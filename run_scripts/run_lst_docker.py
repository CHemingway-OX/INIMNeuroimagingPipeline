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
from annotate import annotate_lesions_fsseg_variablethresh
from utils.utils import getSessionID, getSubjectID, split_list, getfileList, availability_check_systempref


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

def process_lst_ai(dirs, n_container, derivatives_dir, bids_dir, clipping, lesion_thresh, remove_temp=False, probmap=False, use_cpu=False, threads=8 , system = 'GH' , new_annotation = True):
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
        List with all subject IDs for which we want to generate LST-AI lesion segmentations
    derivatives_dir : str
        Path of the LST-AI derivatives folder in the BIDS database
    remove_temp : bool
        Boolean variable indicating if the temp folder should be removed after segmentation files were generated
    use_cpu : bool
        Boolean variable indicating if CPU or GPU should be used for processing

    system : str
        System on which the script is run, e.g., 'GH' for the clinic or 'BMC' for the BMC server.
    
    Returns:
    --------
    None 
        This function produces LST-AI lesion segmentation files
    """
    # iterate through all subject folders
    if system == 'GH':
        print(f'{datetime.datetime.now()} Running LST-AI lesion segmentation on {len(dirs)} subjects on GH system...')
    elif system == 'BMC':
        print(f'{datetime.datetime.now()} Running LST-AI lesion segmentation on {len(dirs)} subjects on BMC system...')
    else:
        raise ValueError(f'Unknown system {system}, please specify system as either "GH" or "BMC".')
    
    for dir in dirs:

        # assemble MPRAGE file lists
        # since we need T1w/MPRAGE AND FLAIR images, we can first list all MPRAGE images and then check if FLAIR image also exists
        if system == 'GH':
            MPRAGE = getfileList(path = dir, 
                                 suffix = '*T1w*')
        elif system == 'BMC':
            MPRAGE = getfileList(path = dir, 
                          suffix = '*T1w*')
        else:
            raise ValueError(f'Unknown system {system}, please specify system as either "GH" or "BMC".')
        MPRAGE = [str(x) for x in MPRAGE if (('.nii.gz' in str(x)) and 
                                       (not 'gadolinium' in str(x)))]
        

        # get subject ID of current subject
        subID = getSubjectID(path = MPRAGE[0])


        # iterate over all sessions with MPRAGE images and check if FLAIR files are available and if lesion segmentation already exists
        for i in range(len(MPRAGE)):
            try:

                # get session ID of current MPRAGE image
                sesID = getSessionID(path = MPRAGE[i])
                # print(str(MPRAGE[i]))
                # check availability of files and folders (create folders if necessary)
                if system == 'GH':
                    flair = str(MPRAGE[i]).replace('_T1w.nii.gz', '_FLAIR.nii.gz')
                elif system == 'BMC':
                    flair = str(MPRAGE[i]).replace('_T1w.nii.gz', '_FLAIR.nii.gz')
                else:
                    raise ValueError(f'Unknown system {system}, please specify system as either "GH" or "BMC".')
                
                if not os.path.exists(flair):
                    raise ValueError(f'sub-{subID}_ses-{sesID}: FLAIR image not available!!')
                
                temp_dir = os.path.join(derivatives_dir, f'sub-{subID}', f'ses-{sesID}', 'temp')
                if not os.path.exists(temp_dir):
                    Path(temp_dir).mkdir(parents=True, exist_ok=True)
                
                deriv_ses = os.path.join(derivatives_dir, f'sub-{subID}', f'ses-{sesID}', 'anat')
                if not os.path.exists(deriv_ses):
                    Path(deriv_ses).mkdir(parents=True, exist_ok=True)
                
                # save derivatives folder in native space for querying later
                deriv_ses_native = deriv_ses
                temp_dir_native = temp_dir
                
                # change directory for docker container processing
                MPRAGE[i] = str(MPRAGE[i]).replace(bids_dir,'/custom_apps/lst_input/')
                flair = str(flair).replace(bids_dir,'/custom_apps/lst_input/')
                temp_dir = str(temp_dir).replace(derivatives_dir,'/custom_apps/lst_output')
                deriv_ses = str(deriv_ses).replace(derivatives_dir,'/custom_apps/lst_output')
                print("temp_dir is ", temp_dir)

                # skip to next case if segmentation already exist
                seg_file = os.path.join(deriv_ses_native, f'sub-{subID}_ses-{sesID}_space-FLAIR_label-lesion_mask.nii.gz')
                seg_file_annot = os.path.join(deriv_ses_native, f'sub-{subID}_ses-{sesID}_space-FLAIR_desc-annotated_label-lesion_mask.nii.gz')
                if os.path.exists(seg_file) and os.path.exists(seg_file_annot):
                    print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: LST-AI lesion segmentation already exists, skip and proceed to next case...')
                    continue
                
                ### modify here:
                # define command line for LST-AI
                if probmap:
                    if remove_temp and use_cpu:
                        command = f'--t1 {MPRAGE[i]} --flair {flair} --output {deriv_ses} --probability_map --device cpu --clipping {clipping[0]} {clipping[1]} --threads {threads} --lesion_threshold {lesion_thresh}'
                    elif remove_temp:
                        command = f'--t1 {MPRAGE[i]} --flair {flair} --output {deriv_ses} --probability_map --device 0 --clipping {clipping[0]} {clipping[1]} --threads {threads} --lesion_threshold {lesion_thresh}'
                    elif use_cpu:
                        command = f'--t1 {MPRAGE[i]} --flair {flair} --output {deriv_ses} --probability_map --temp {temp_dir} --device cpu --clipping {clipping[0]} {clipping[1]} --threads {threads} --lesion_threshold {lesion_thresh}'
                    else:
                        command = f'--t1 {MPRAGE[i]} --flair {flair} --output {deriv_ses} --probability_map --temp {temp_dir} --device 0 --clipping {clipping[0]} {clipping[1]} --threads {threads} --lesion_threshold {lesion_thresh}'
                else:
                    if remove_temp and use_cpu:
                        command = f'--t1 {MPRAGE[i]} --flair {flair} --output {deriv_ses} --device cpu --clipping {clipping[0]} {clipping[1]} --threads {threads} --lesion_threshold {lesion_thresh}'
                    elif remove_temp:
                        command = f'--t1 {MPRAGE[i]} --flair {flair} --output {deriv_ses} --device 0 --clipping {clipping[0]} {clipping[1]} --threads {threads} --lesion_threshold {lesion_thresh}'
                    elif use_cpu:
                        command = f'--t1 {MPRAGE[i]} --flair {flair} --output {deriv_ses} --temp {temp_dir} --device cpu --clipping {clipping[0]} {clipping[1]} --threads {threads} --lesion_threshold {lesion_thresh}'
                    else:
                        command = f'--t1 {MPRAGE[i]} --flair {flair} --output {deriv_ses} --temp {temp_dir} --device 0 --clipping {clipping[0]} {clipping[1]} --threads {threads} --lesion_threshold {lesion_thresh}'
                # print(command)
                # run LST-AI
                
                docker_exec = f"docker exec container_{n_container} lst "

                command2 = " ".join([docker_exec,command])

                print(command2)
                subprocess.run(command2, shell=True)

                # check if folder contains *seg-lst.nii.gz files, indicating that LST-AI successfully finished, and rename files
                output_anat_files = os.listdir(deriv_ses_native)
                les_vol_file = os.path.join(deriv_ses_native, f'sub-{subID}_ses-{sesID}_lesion_stats.csv')
                les_vol_annot_file = os.path.join(deriv_ses_native, f'sub-{subID}_ses-{sesID}_annotated_lesion_stats.csv')
                
                if ('space-flair_seg-lst.nii.gz' in output_anat_files) and ('space-flair_desc-annotated_seg-lst.nii.gz' in output_anat_files):
                    
                    print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: Rename LST-AI lesion mask (BIDS)...')
                    os.rename(os.path.join(deriv_ses_native,'space-flair_seg-lst.nii.gz'), seg_file)
                    os.rename(os.path.join(deriv_ses_native,'space-flair_desc-annotated_seg-lst.nii.gz'), seg_file_annot)
                    os.rename(os.path.join(deriv_ses_native,'lesion_stats.csv'), les_vol_file)
                    os.rename(os.path.join(deriv_ses_native,'annotated_lesion_stats.csv'), les_vol_annot_file)
                    if os.path.exists(seg_file) and os.path.exists(seg_file_annot):
                        print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: Rename LST-AI lesion mask (BIDS) DONE!')

                    # also rename auxiliary files if they are available
                    output_temp_files = os.listdir(temp_dir_native)
                    if (len(output_temp_files)>0):
                        print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: Rename LST-AI auxiliary files (BIDS)...')
                        # iterate over all output files and rename to BIDS
                        for filename in output_temp_files:
                            if ('sub-X_ses-Y' in filename):
                                os.rename(os.path.join(temp_dir_native, filename), os.path.join(temp_dir_native, str(filename).replace('sub-X_ses-Y', f'sub-{subID}_ses-{sesID}')))
                            else:
                                os.rename(os.path.join(temp_dir_native, filename), os.path.join(temp_dir_native, f'sub-{subID}_ses-{sesID}_{filename}'))
                        print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: Rename LST-AI auxiliary files (BIDS) DONE!')

                    if new_annotation:
                        # prob_out = os.path.join(temp_dir_native + f'/temp/sub-{subID}_ses-{sesID}_space-FLAIR_seg-lst_prob.nii.gz')
                        # fs_seg = test_bids_dir + f'derivatives/fastsurfer_v2.4.2_docker/' + f'sub-{subID}/ses-{sesID}/sub-{subID}_ses-{sesID}/mri/aparc.DKTatlas+aseg.mapped.mgz'
                        # out_annotated_native = os.path.join(deriv_ses_native, f"sub-{subID}/ses-{sesID}/temp/sub-{subID}_ses-{sesID}_space-flair_desc-annotated_fsseg.nii.gz")
                        # assert prob_out and fs_seg and out_annotated_native
                        # annotate_lesions_fsseg_variablethresh(prob_out, fs_seg , out_annotated_native)

                        print('performing new annotation ...')
                        prob_out = os.path.join(temp_dir_native , f'sub-{subID}_ses-{sesID}_space-FLAIR_seg-lst_prob.nii.gz')
                        fs_seg = os.path.join(bids_dir + f'/derivatives/fastsurfer_v2.4.2_docker/' + f'sub-{subID}/ses-{sesID}/sub-{subID}_ses-{sesID}/mri/aparc.DKTatlas+aseg.mapped.mgz')
                        out_annotated_native = os.path.join(temp_dir_native, f"sub-{subID}_ses-{sesID}_space-flair_desc-annotated_fsseg.nii.gz")
                        
                        print(prob_out , fs_seg , out_annotated_native)
                        annotate_lesions_fsseg_variablethresh(prob_out, fs_seg , out_annotated_native)

                else:
                    print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: failed to generate segmentation, delete ouput folder...')
                    shutil.rmtree(str(Path(deriv_ses_native).parent))
                    if os.path.exists(deriv_ses_native) or os.path.exists(temp_dir_native):
                        raise ValueError(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: failed to delete the derivatives folder(s)!')
                    else:
                        print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: successfully deleted the derivatives folder(s)!')
                        continue

            except:
                print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: Error occured during processing, proceeding with next case.')

    docker_stop = f"docker stop container_{n_container}"
    docker_rm = f"docker rm container_{n_container}"

    subprocess.run(docker_stop, shell=True)
    subprocess.run(docker_rm, shell=True)

#### modify here            
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Run LST-AI Pipeline on cohort.')

    parser.add_argument('-i', '--input_directory', 
                        help='BIDS database.', 
                        required=True)

    parser.add_argument('-n', '--number_of_workers', 
                        help='Number of parallel processing cores.', 
                        type=int, 
                        default=1)
    
    parser.add_argument('-t', '--threads', 
                        help='Number of parallel processing cores assigned to one worker.', 
                        type=int, 
                        default=8)

    parser.add_argument('--cpu', 
                        help='Use the --cpu flag if you only want to use CPU.', 
                        action='store_true')
    
    parser.add_argument('--remove_temp', 
                        help='Use the --remove_temp flag if you want to remove the temp folder containing auxiliary files.', 
                        action='store_true')
    
    parser.add_argument('--probability_map', 
                        dest='probability_map',
                        help='Use the --probability-map flag if you want to generate probability maps.', 
                        action='store_true',
                        default=True)

    parser.add_argument('--clipping',
                        dest='clipping',
                        help='Clipping for standardization of image intensities (default: (min,max)=(0.5,99.5))',
                        nargs='+',
                        type=float,
                        default=(0.5, 99.5))
    
    parser.add_argument('--lesion_threshold',
                        dest='lesion_threshold',
                        help='Minimum lesion volume threshold',
                        type=int,
                        default=0)
    
    parser.add_argument('--system',
                        dest='system',
                        help='System on which the script is run, e.g., "GH" for the clinic or "BMC" for the BMC server.',
                        type=str,
                        default='GH')
    
    parser.add_argument('--new_annotation',
                        dest='new_annotation',
                        help='Whether to perform new annotation based on FastSurfer segmentation.',
                        action='store_true',
                        default=True)
    parser.add_argument('--subjects',
                        help='Comma-separated subject IDs to process, e.g. 001,002,sub-010.',
                        type=str,
                        default=None)

    
    # parser.add_argument('--new_annotation',
    #                     type=bool,
    #                     default=True)
    
    # read the arguments
    args = parser.parse_args()

    if args.cpu:
        use_cpu = True
    else:
        use_cpu = False

    if args.remove_temp:
        remove_temp = True
    else:
        remove_temp = False

    # if args.new_annotation:
    #     new_annotation = True
    # else:
    #     new_annotation = False
    
    input_path = args.input_directory
    # n_workers = args.number_of_workers # define the number of container instances created
    n_workers = args.number_of_workers if args.number_of_workers > 0 else 1 # ensure at least one worker is used
    # generate derivatives #### modify here?
    derivatives_dir = os.path.join(input_path, "derivatives/lst-ai-v1.2.0_docker")
    if not os.path.exists(derivatives_dir):
        Path(derivatives_dir).mkdir(parents=True, exist_ok=True)
    print(f"deriv dir is {derivatives_dir}")
    # generate list with subject folders for multiprocessing
    data_root = Path(os.path.join(input_path))
    dirs = sorted(list(data_root.glob('*')))
    dirs = [str(x) for x in dirs]
    dirs = [x for x in dirs if "sub-" in x]
    dirs = filter_subject_dirs(dirs, parse_subject_ids(args.subjects))
    
    # check which files have already been processed
    dirs_missing, dirs_processed = availability_check_systempref(sub_dirs=dirs,
                                                      deriv_dir=derivatives_dir,
                                                      file_suffix='space-FLAIR_label-lesion_mask.nii.gz',
                                                      system=args.system)
    print(f'Number of incomplete subjects: {len(dirs_missing)}')
    print(f'Number of complete subjects: {len(dirs_processed)}')
    print(dirs_missing)
    
    # only split the list of subjects with missing LST-AI lesion segmentation for multiprocessing
    files = split_list(alist = dirs_missing, 
                       splits = n_workers)

    # initialize multithreading
    pool = multiprocessing.Pool(processes=n_workers)
    # call samseg processing function in multiprocessing setting
    # for testing: export BIDS='/mnt/d/TWIN_MRI/BIDS_directories/TEST'
    print(derivatives_dir)
    for x in range(0, n_workers):
        docker_run = f"docker run -id --name container_{x} --user root --gpus all"
        v_in = "".join([" -v " , input_path , ":/custom_apps/lst_input"])
        v_out = "".join([" -v " , derivatives_dir , ":/custom_apps/lst_output"])
        other_input = " --entrypoint /bin/bash jqmcginnis/lst-ai:v1.2.0"

        docker_run_command = " ".join([docker_run + v_in + v_out + other_input])
        print(docker_run_command)
        subprocess.run(docker_run_command, shell=True)

    for x in range(0, n_workers):
        pool.apply_async(process_lst_ai, args=(files[x], x,
                                               derivatives_dir, 
                                               input_path,
                                               args.clipping, 
                                               args.lesion_threshold, 
                                               remove_temp, 
                                               args.probability_map, 
                                               use_cpu, 
                                               args.threads,
                                               args.system, 
                                               args.new_annotation))
        
    pool.close()
    pool.join()

    print('DONE!')
