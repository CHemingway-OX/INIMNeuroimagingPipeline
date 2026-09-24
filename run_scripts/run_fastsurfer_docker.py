# copied from TWILTGEN/LST_AI_BIDS on 25.04.2025
# modified to suit other analysis methods other than LST_AI
# step 1: implement LST_AI in Docker File.

import argparse
import os
import sys                                                                                                                                                                       
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))                                                                                 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
import shutil
import datetime
import subprocess
from pathlib import Path
import multiprocessing
from utils.utils import getSessionID, getSubjectID, split_list, getfileList, availability_check_fastsurfer


def parse_subject_ids(subjects_arg):
    if not subjects_arg:
        return None
    subject_ids = set()
    for item in subjects_arg.split(","):
        item = item.strip()
        if not item:
            continue
        subject_ids.add(item.replace("sub-", ""))
    return subject_ids or None


def filter_subject_dirs(dirs, subject_ids):
    if subject_ids is None:
        return dirs
    return [x for x in dirs if getSubjectID(x) in subject_ids]

def process_fastsurfer(dirs, n_container, derivatives_dir, bids_dir, pipeline_arg, use_cpu=False, threads=8 , system = 'GH' , longitudinal=False):
    """
    This function applies LST-AI lesion segmentation and also applies required pre-processing steps of the MPRAGE and FLAIR images. 
    Pre-processing includes skull-stripping and image registration. 
    We use the original MPRAGE and FLAIR images as input.
    Next, we check if SAMSEG segmentation was successful by making sure that the space-orig_seg-lst.nii.gz file was generated.
    All resulting files are saved to a temp folder and the segmentation files are save to anat folderin derivatives. 
    In order to be compliant with BIDS convention, we rename the output files. 
    Optionally, the output folder can be deleted (e.g., to clean up if it is not needed anymore)

    Parameters:
    -----------
    dirs : list
        List with all subject IDs for which we want to generate FastSurfer processing
    derivatives_dir : str
        Path of the LST-AI derivatives folder in the BIDS database
    use_cpu : bool
        Boolean variable indicating if CPU or GPU should be used for processing

    system : str
        System on which the script is run, e.g., 'GH' for the clinic or 'BMC' for the BMC server.
    
    Returns:
    --------
    None 
        This function executes the FastSurfer processing and saves the results to the derivatives folder.
    """
    # check system
    if system == 'GH':
        print(f'{datetime.datetime.now()} Running FastSurfer processing on {len(dirs)} subjects on GH system...')
    elif system == 'BMC':
        print(f'{datetime.datetime.now()} Running FastSurfer processing on {len(dirs)} subjects on BMC system...')
    else:
        raise ValueError(f'Unknown system {system}, please specify system as either "GH" or "BMC".')
    
    # iterate through all subject folders
    for dir in dirs:

        # assemble MPRAGE file lists
        # since we need T1w/MPRAGE; T2w are optional for Hypthalamus segmentation
        if system == 'GH':
            MPRAGE = getfileList(path = dir, 
                                 suffix = '*T1w*')
        elif system == 'BMC':
            MPRAGE = getfileList(path = dir, 
                          suffix = '*T1w*')
        else:
            raise ValueError(f'Unknown system {system}, please specify system as either "GH" or "BMC".')
        
        if "LIT" in dir:
            MPRAGE = getfileList(path = dir, 
                                 suffix = '*inpainting_result.nii.gz')
        else: 
            MPRAGE = [str(x) for x in MPRAGE if (('.nii.gz' in str(x)) and 
                                       (not 'gadolinium' in str(x)))]
        
        

        # get subject ID of current subject
        subID = getSubjectID(path = MPRAGE[0])


        # iterate over all sessions with MPRAGE images and check if FLAIR files are available and if segmentation AND/OR surface reconstruction already exists
        for i in range(len(MPRAGE)):
            # try:
                # get session ID of current MPRAGE image
                sesID = getSessionID(path = MPRAGE[i])

                # check availability of files and folders (create folders if necessary)
                if 'no_hypothal' not in pipeline_arg and 'LIT' not in dir:
                    if system == 'GH':
                        flair = str(MPRAGE[i]).replace('_MPRAGE.nii.gz', '_FLAIR.nii.gz')
                    elif system == 'BMC':
                        flair = str(MPRAGE[i]).replace('_T1w.nii.gz', '_FLAIR.nii.gz')
                    else:
                        raise ValueError(f'Unknown system {system}, please specify system as either "GH" or "BMC".')
                    
                    if not os.path.exists(flair):
                        raise ValueError(f'sub-{subID}_ses-{sesID}: FLAIR image not available!!')
                
                deriv_ses = os.path.join(derivatives_dir, f'sub-{subID}', f'ses-{sesID}','anat')
                if not os.path.exists(deriv_ses):
                    Path(deriv_ses).mkdir(parents=True, exist_ok=True)

                # if '--surf_only' in pipeline_arg:
                #     # check if asegdkt segmentation file is available
                #     asegdkt_segfile = os.path.join(deriv_ses , f'sub-{subID}_ses-{sesID}', 'mri', 'aparc.DKTatlas+aseg.deep.mgz')
                #     if not os.path.exists(asegdkt_segfile):
                #         raise ValueError(f'sub-{subID}_ses-{sesID}: asegdkt segmentation file not available!! Please specify --asegdkt_segfile argument with the path to the asegdkt segmentation file.')
                

                # save derivatives folder in native space for querying later
                deriv_ses_native = deriv_ses
                subject_deriv = os.path.join(deriv_ses_native, f'sub-{subID}_ses-{sesID}')
                seg_output = os.path.join(subject_deriv, 'mri', 'aparc.DKTatlas+aseg.deep.mgz')
                lh_surf_output = os.path.join(subject_deriv, 'surf', 'lh.pial')
                rh_surf_output = os.path.join(subject_deriv, 'surf', 'rh.pial')

                if 'seg_only' in pipeline_arg:
                    already_done = os.path.exists(seg_output)
                elif 'surf_only' in pipeline_arg:
                    already_done = os.path.exists(lh_surf_output) and os.path.exists(rh_surf_output)
                else:
                    already_done = (
                        os.path.exists(seg_output)
                        and os.path.exists(lh_surf_output)
                        and os.path.exists(rh_surf_output)
                    )

                if already_done:
                    print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: requested FastSurfer outputs already exist, skip and proceed to next case...')
                    continue

                # change directory for docker container processing
                MPRAGE[i] = str(MPRAGE[i]).replace(bids_dir,'/data')
                if ('no_hypothal' not in pipeline_arg) and ('LIT' not in dir):
                    flair = str(flair).replace(bids_dir,'/data')
                deriv_ses = str(deriv_ses).replace(derivatives_dir,'/output/')

                base_command = f'./run_fastsurfer.sh --sid sub-{subID}_ses-{sesID} --sd /output/sub-{subID}/ses-{sesID} --t1 {MPRAGE[i]} --fs_license /fs_license/license.txt'
                additional_args = f''
                print(f'Pipeline arguments: {pipeline_arg}')
        
                if 'surf_only' in pipeline_arg:
                    additional_args += f' --asegdkt_segfile {deriv_ses}/mri/aparc.DKTatlas+aseg.deep.mgz --surf_only'
                if 'seg_only' in pipeline_arg:
                    additional_args += ' --seg_only'
                if 'no_cereb' in pipeline_arg:
                    additional_args += ' --no_cereb'
                if '--no_hypothal ' in pipeline_arg:
                    additional_args += ' --no_hypothal'
                if 'no_hypothal' not in pipeline_arg and 'LIT' not in dir:
                    additional_args += f' --t2 {flair}'
                # resource options
                if use_cpu:
                    additional_args += f' --cpu'
                additional_args += f' --3T --threads {threads}'

                
                docker_exec = f"docker exec fastsurfer_{n_container} "

                command2 = " ".join([docker_exec,base_command, additional_args])
                print(additional_args)
                print(command2)

                subprocess.run(command2, shell=True)

                # check if folder contains segmentation and surface reconstruction results, indicating that Fastsurfer successfully finished, and rename files
                output_anat_files = os.listdir(deriv_ses_native)
                if os.path.exists(seg_output) and os.path.exists(lh_surf_output) and os.path.exists(rh_surf_output):
                    print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: FastSurfer processing DONE!')

                else:
                    print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: FastSurfer processing FAILED!')

    docker_stop = f"docker stop fastsurfer_{n_container}"
    docker_rm = f"docker rm fastsurfer_{n_container}"

    subprocess.run(docker_stop, shell=True)
    subprocess.run(docker_rm, shell=True)

#### modify here            
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Run FastSurfer on BIDS dataset.')

    parser.add_argument('-i', '--input_directory', 
                        help='BIDS database.', 
                        required=True)

    parser.add_argument('--fs_license',
                        help='Path to the FreeSurfer license file.', 
                        type=str, 
                        default='/home/chemingway/freesurfer/'
                        )
    
    parser.add_argument('--do_segmentation',
                        help='Use the --do_segmentation flag if you want to run fast surfer segmentation.', 
                        action='store_true',
                        default=True)

    parser.add_argument('--no_cereb',
                        help='Use the --no_cereb flag if you do not want to run cerebellum segmentation.', 
                        action='store_true',
                        default=False)

    parser.add_argument('--no_hypothal',
                        help='Use the --no_hypothal flag if you do not want to run hypothalamus segmentation.', 
                        action='store_true',
                        default=True)
    
    parser.add_argument('--do_recon',
                        help='Use the --do_recon flag if you want to run cortical surface reconstruction (recon-surf), requires a segmentation specified in the derivatives folder.', 
                        action='store_true',
                        default=True)
    parser.add_argument('--asegdkt_segfile',
                        help='Path to the asegdkt segmentation file, required for cortical surface reconstruction (recon-surf), if prior segmentation exists.', 
                        type=str, 
                        default=None)
    # resource options:
    parser.add_argument('-t', '--threads', 
                        help='Number of parallel processing cores assigned to one worker.', 
                        type=int, 
                        default=8)

    parser.add_argument('--cpu', 
                        help='Use the --cpu flag if you only want to use CPU.', 
                        action='store_true')

    parser.add_argument('--system',
                        dest='system',
                        help='System on which the script is run, e.g., "GH" for the clinic or "BMC" for the BMC server.',
                        type=str,
                        default='GH')
    parser.add_argument('--longitudinal',
                        help='Use the --longitudinal flag if you want to run FastSurfer additionally on longitudinal data.',
                        action='store_true',
                        default=False)
    parser.add_argument('--lit',
                        help="Use the --lit flag if the inputs have been inpainted with FS-LIT.",
                        action='store_true',
                        default=False)
    parser.add_argument('--subjects',
                        help='Comma-separated subject IDs to process, e.g. 001,002,sub-010.',
                        type=str,
                        default=None)

    
    # read the arguments
    args = parser.parse_args()

    if args.cpu:
        use_cpu = True
    else:
        use_cpu = False

    if args.do_segmentation and args.do_recon:
        pipeline_arg = ['']
        if args.no_cereb:
            pipeline_arg.append('--no_cereb ')
        if args.no_hypothal:
            pipeline_arg.append('--no_hypothal ')
    elif args.do_segmentation and not args.do_recon:
        pipeline_arg = ['--seg_only']
        if args.no_cereb:
            pipeline_arg.append(' --no_cereb ')
        if args.no_hypothal:
            pipeline_arg.append('--no_hypothal ')
    elif not args.do_segmentation and args.do_recon:
        pipeline_arg = ['--surf_only']
        if args.asegdkt_segfile is None:
            raise ValueError('You need to specify the asegdkt segmentation file with --asegdkt_segfile if you want to run cortical surface reconstruction (recon-surf)!')
        else:
            pipeline_arg.append(f'--asegdkt_segfile {args.asegdkt_segfile}')
        if args.no_cereb or args.no_hypothal:
            raise ValueError('You cannot use the --no_cereb or --no_hypothal flags if you only want to run cortical surface reconstruction (recon-surf)!')
    else:
        raise ValueError('You need to specify at least one of --do_segmentation or --do_recon, or both!')
        
    print(f'Pipeline arguments: {pipeline_arg}')

    # if args.lit:
        

    input_path = args.input_directory
    
    fs_license = args.fs_license
    # n_workers = args.number_of_workers # define the number of container instances created
    n_workers =  1 # ensure at least one worker is used
    # generate derivatives #### modify here?
    if not args.lit:
        derivatives_dir = os.path.join(input_path, "derivatives/fastsurfer_v2.4.2_docker/")
        print(f"deriv dir is {derivatives_dir}")
        try:
            Path(derivatives_dir).mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            # Directory already exists, possibly created by another process
            pass
    
    if args.lit:
        derivatives_dir = os.path.join(input_path, "derivatives/fastsurfer_v2.4.2_docker_FS-LIT/")
        print(f"deriv dir is {derivatives_dir}")
        try:
            Path(derivatives_dir).mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            # Directory already exists, possibly created by another process
            pass
    
    if args.lit:
        input_path = os.path.join(input_path , "derivatives" , "FS-LIT2")

    # generate list with subject folders for multiprocessing
    data_root = Path(os.path.join(input_path))
    dirs = sorted(list(data_root.glob('*')))
    dirs = [str(x) for x in dirs]
    dirs = [x for x in dirs if "sub-" in x]
    dirs = filter_subject_dirs(dirs, parse_subject_ids(args.subjects))
    # print(dirs)
    # check which files have already been processed

    dirs_missing, dirs_processed = availability_check_fastsurfer(sub_dirs=dirs,
                                                      deriv_dir=derivatives_dir,
                                                      file_suffix=['mri/aparc.DKTatlas+aseg.deep.mgz',
                                                                   'surf/lh.pial',
                                                                   'surf/rh.pial'],
                                                      system=args.system)
    print(f'Number of incomplete subjects: {len(dirs_missing)}')
    print(f'Number of complete subjects: {len(dirs_processed)}')
    print(dirs_missing)

    # only split the list of subjects with missing LST-AI lesion segmentation for multiprocessing
    files = split_list(alist = dirs_missing, 
                       splits = n_workers)

    # initialize multithreading
    pool = multiprocessing.Pool(processes=1)
    
    print(files)
    x = 0
    docker_run = f"docker run -id --name fastsurfer_{x} -u $(id -u):$(id -g) --gpus all"
    v_in = "".join([" -v " , input_path , ":/data:ro"])
    v_out = "".join([" -v " , derivatives_dir , ":/output"])
    v_license = "".join([" -v ", fs_license , ":/fs_license"])
    other_input = " --entrypoint /bin/bash deepmi/fastsurfer:latest"

    docker_run_command = " ".join([docker_run + v_in + v_out + v_license + other_input])
    print(docker_run_command)
    subprocess.run(docker_run_command, shell=True)

    print(f'{datetime.datetime.now()} Starting FastSurfer processing on {len(files[x])} subjects...')
    # pool.apply_async(process_fastsurfer, args=(files[x], x,
    #                                            derivatives_dir, 
    #                                            input_path,
    #                                            pipeline_arg, 
                                                
    #                                            use_cpu, 
    #                                            args.threads,
    #                                            args.system,
    #                                            args.longitudinal))
    
    process_fastsurfer(files[x], x,
                       derivatives_dir, 
                       input_path,
                       pipeline_arg, 
                       use_cpu, 
                       args.threads,
                       args.system,
                       args.longitudinal)

    # pool.close()
    # pool.join()

    print('DONE!')
