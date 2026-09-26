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
from utils.container_runtime import run_container
from annotate import annotate_lesions_fsseg_variablethresh
from utils.utils import getSessionID, getSubjectID, split_list, getfileList, availability_check_systempref

# HD-BET (inside LST-AI) loads its weights from ~/hd-bet_params. The image keeps
# them in /root with mode 600, which a non-root Singularity user cannot read.
HDBET_CONTAINER_HOME = "/hdbet_home"


def hdbet_home_dir():
    default = Path(os.environ.get("CONTAINER_DIR", "containers")).expanduser().resolve().parent / "models" / "hdbet_home"
    home = Path(os.environ.get("HDBET_HOME", str(default))).expanduser().resolve()
    missing = [n for n in range(5) if not (home / "hd-bet_params" / f"{n}.model").is_file()]
    if missing:
        raise FileNotFoundError(
            f"HD-BET weights missing in {home}/hd-bet_params (folds {missing}). "
            "Set HDBET_HOME to a directory containing hd-bet_params/0.model..4.model."
        )
    return home


def fastsurfer_seg_path(bids_dir, subID, sesID):
    fastsurfer_dir = os.environ.get('FASTSURFER_OUTPUT_DIR', os.path.join(bids_dir, 'derivatives', 'fastsurfer_v2.4.2_docker'))
    return os.path.join(fastsurfer_dir, f'sub-{subID}', f'ses-{sesID}', f'sub-{subID}_ses-{sesID}', 'mri', 'aparc.DKTatlas+aseg.mapped.mgz')


def annotation_path(temp_dir_native, subID, sesID):
    return os.path.join(temp_dir_native, f'sub-{subID}_ses-{sesID}_space-flair_desc-annotated_fsseg.nii.gz')


def missing_fastsurfer_segs(dirs, bids_dir, derivatives_dir):
    """FastSurfer segmentations needed for annotation but absent, so failures surface before LST-AI runs."""
    missing = []
    for dir in dirs:
        for t1w in getfileList(path=dir, suffix='*T1w*'):
            t1w = str(t1w)
            if '.nii.gz' not in t1w or 'gadolinium' in t1w:
                continue
            if not os.path.exists(t1w.replace('_T1w.nii.gz', '_FLAIR.nii.gz')):
                continue
            subID, sesID = getSubjectID(path=t1w), getSessionID(path=t1w)
            temp_dir_native = os.path.join(derivatives_dir, f'sub-{subID}', f'ses-{sesID}', 'temp')
            fs_seg = fastsurfer_seg_path(bids_dir, subID, sesID)
            if not os.path.isfile(annotation_path(temp_dir_native, subID, sesID)) and not os.path.isfile(fs_seg):
                missing.append(fs_seg)
    return missing


def run_annotation(temp_dir_native, bids_dir, subID, sesID):
    print('performing new annotation ...')
    prob_out = os.path.join(temp_dir_native, f'sub-{subID}_ses-{sesID}_space-FLAIR_seg-lst_prob.nii.gz')
    fs_seg = fastsurfer_seg_path(bids_dir, subID, sesID)
    out_annotated_native = annotation_path(temp_dir_native, subID, sesID)
    print(prob_out, fs_seg, out_annotated_native)
    annotate_lesions_fsseg_variablethresh(prob_out, fs_seg, out_annotated_native)


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

def process_lst_ai(dirs, n_container, derivatives_dir, bids_dir, clipping, lesion_thresh, remove_temp=False, probmap=False, use_cpu=False, threads=8 , system = 'GH' , new_annotation = True, annotation_only = False):
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


        if not MPRAGE:
            raise ValueError(f'No non-contrast T1w images found in {dir}')
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
                annotation_done = (not new_annotation or os.path.isfile(annotation_path(temp_dir_native, subID, sesID)))
                lst_done = os.path.exists(seg_file) and os.path.exists(seg_file_annot)
                if lst_done and annotation_done:
                    print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: LST-AI lesion segmentation already exists, skip and proceed to next case...')
                    continue
                prob_map = os.path.join(temp_dir_native, f'sub-{subID}_ses-{sesID}_space-FLAIR_seg-lst_prob.nii.gz')
                if annotation_only and not (lst_done and os.path.isfile(prob_map)):
                    raise RuntimeError(f'sub-{subID}_ses-{sesID}: LST-AI outputs or probability map missing; '
                                       'run the LST-AI segmentation (GPU stage) first')
                if lst_done and os.path.isfile(prob_map):
                    print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: LST-AI outputs exist, running only the missing annotation...')
                    run_annotation(temp_dir_native, bids_dir, subID, sesID)
                    continue

                command = ["lst", "--t1", MPRAGE[i], "--flair", flair,
                           "--output", deriv_ses, "--device", "cpu" if use_cpu else "0",
                           "--clipping", str(clipping[0]), str(clipping[1]),
                           "--threads", str(threads), "--lesion_threshold", str(lesion_thresh)]
                if probmap:
                    command += ["--probability_map"]
                if not remove_temp:
                    command += ["--temp", temp_dir]
                run_container("lst", command, [
                    f"{Path(bids_dir).resolve()}:/custom_apps/lst_input:ro",
                    f"{Path(derivatives_dir).resolve()}:/custom_apps/lst_output",
                    f"{hdbet_home_dir()}:{HDBET_CONTAINER_HOME}:ro",
                ], gpu=not use_cpu, home=HDBET_CONTAINER_HOME)

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
                            if filename.startswith(f'sub-{subID}_ses-{sesID}_'):
                                continue
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

                        run_annotation(temp_dir_native, bids_dir, subID, sesID)

                else:
                    raise RuntimeError(f'sub-{subID}_ses-{sesID}: LST-AI outputs are incomplete; intermediate files retained')
            except Exception:
                print(f'Processing failed for {MPRAGE[i]}', file=sys.stderr)
                raise

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
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument('--skip_annotation',
                       help='Segment lesions only; the FastSurfer-based annotation needs surfaces (CPU stage).',
                       action='store_true')
    stage.add_argument('--annotation_only',
                       help='Only annotate existing LST-AI outputs; starts no container.',
                       action='store_true')


    # parser.add_argument('--new_annotation',
    #                     type=bool,
    #                     default=True)

    # read the arguments
    args = parser.parse_args()
    if args.skip_annotation:
        args.new_annotation = False
    if not args.annotation_only:
        hdbet_home_dir()  # fail before any container starts

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

    input_path = str(Path(args.input_directory).resolve())
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

    if args.new_annotation:
        missing = missing_fastsurfer_segs(dirs, input_path, derivatives_dir)
        if missing:
            raise SystemExit(
                'FastSurfer segmentation required for lesion annotation is missing:\n  '
                + '\n  '.join(missing)
                + '\nRun FastSurfer first. If FastSurfer ran longitudinally, pass --long to struct_bids.sh '
                  '(or set FASTSURFER_OUTPUT_DIR).'
            )

    dirs_missing = dirs

    # only split the list of subjects with missing LST-AI lesion segmentation for multiprocessing
    files = split_list(alist = dirs_missing,
                       splits = n_workers)

    if n_workers == 1:
        process_lst_ai(files[0], 0, derivatives_dir, input_path, args.clipping,
                       args.lesion_threshold, remove_temp, args.probability_map,
                       use_cpu, args.threads, args.system, args.new_annotation,
                       args.annotation_only)
    else:
        with multiprocessing.Pool(processes=n_workers) as pool:
            results = [pool.apply_async(process_lst_ai, args=(batch, x, derivatives_dir,
                       input_path, args.clipping, args.lesion_threshold, remove_temp,
                       args.probability_map, use_cpu, args.threads, args.system,
                       args.new_annotation, args.annotation_only)) for x, batch in enumerate(files)]
            for result in results:
                result.get()  # Propagate worker failures to the shell and Slurm.
    print('DONE!')
