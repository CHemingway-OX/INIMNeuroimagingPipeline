# copied from TWILTGEN/LST_AI_BIDS on 25.04.2025
# modified to suit other analysis methods other than LST_AI

import argparse
import os
import sys                                                                                                                                                                       
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))                                                                                 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))        
import datetime
import subprocess
from pathlib import Path
import re
import ants
from utils.utils import getSessionID, getSubjectID, split_list, getfileList, availability_check
import numpy as np
from utils.container_runtime import run_container


def run_lit_container(input_image: str, mask_image: str, output_directory: str, dilate: int) -> None:
    input_image = str(Path(input_image).resolve())
    mask_image = str(Path(mask_image).resolve())
    output_directory = str(Path(output_directory).resolve())
    binds = [f"{input_image}:{input_image}:ro", f"{mask_image}:{mask_image}:ro",
             f"{output_directory}:{output_directory}"]
    # Packaged code and weights are used unless an explicit override is requested.
    if os.environ.get("LIT_REPO"):
        repo = Path(os.environ["LIT_REPO"]).expanduser().resolve()
        required = [repo / "run_lit.sh", repo / "lit/inpaint_image.py"]
        required += [repo / "weights" / f"model_{view}.pt" for view in ("axial", "coronal", "sagittal")]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("LIT_REPO override is missing code/weights: " + ", ".join(missing))
        binds.append(f"{repo}:/inpainting:ro")
    run_container("lit", ["/bin/bash", "/inpainting/run_lit.sh", "-i", input_image,
                  "-m", mask_image, "-o", output_directory, "--dilate", str(dilate)],
                  binds, gpu=True, workdir="/inpainting")


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


def process_LIT(dirs, LST_dir, out_dir, dilate = 1):
    """
    This function implements Fastsurfer LIT
    """
    print("starting2")
    # iterate through all subject folders
    for dir in dirs:
        # assemble MPRAGE file lists
        # since we need MPRAGE AND FLAIR images, we can first list all MPRAGE images and then check if FLAIR image also exists
        MPRAGE = getfileList(path = dir, 
                          suffix = '*T1w*')
        MPRAGE = [str(x) for x in MPRAGE if (('.nii.gz' in str(x)) and 
                                       (not 'gadolinium' in str(x)))]

        # get subject ID of current subject
        subID = getSubjectID(path = MPRAGE[0])

        # iterate over all sessions with MPRAGE images and check if lesion inpainting already exists
        for i in range(len(MPRAGE)):
            # try:
                # get session ID of current MPRAGE image
                sesID = getSessionID(path = MPRAGE[i])
                print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: Processing lesion inpainting with FS-LIT...')
                # print(str(MPRAGE[i]))
                # check availability of files and folders (create folders if necessary)
                
                temp_dir = os.path.join(out_dir, f'sub-{subID}', f'ses-{sesID}', 'temp')
                if not os.path.exists(temp_dir):
                    Path(temp_dir).mkdir(parents=True, exist_ok=True)
                
                deriv_ses = os.path.join(out_dir, f'sub-{subID}', f'ses-{sesID}', 'anat')
                if not os.path.exists(deriv_ses):
                    Path(deriv_ses).mkdir(parents=True, exist_ok=True)
                # lesion mask image from LST-AI in flair space
                mask_image_space_flair = os.path.join(
                    LST_dir, f'sub-{subID}', f'ses-{sesID}', 'anat', f'sub-{subID}_ses-{sesID}_space-FLAIR_label-lesion_mask.nii.gz')
                moving=ants.image_read(mask_image_space_flair)
                
                # --- Transform 1: flair to mni ---
                transform_flair_to_mni_txt_path = os.path.join(
                    LST_dir, f'sub-{subID}', f'ses-{sesID}', 'temp', f'sub-{subID}_ses-{sesID}_affine_flair_to_mni.mat')
                transform_flair_to_mni_mat = np.loadtxt(transform_flair_to_mni_txt_path)
                transform_flair_to_mni_obj = ants.create_ants_transform(transform_type='AffineTransform', dimension=3)
                transform_flair_to_mni_obj.set_parameters(list(transform_flair_to_mni_mat[:3,:3].ravel()) + list(transform_flair_to_mni_mat[:3,3]))
                
                transform_flair_to_mni_ants_path = os.path.join(temp_dir, f'sub-{subID}_ses-{sesID}_flair_to_mni_ants.mat')
                ants.write_transform(transform_flair_to_mni_obj, transform_flair_to_mni_ants_path)

                # --- Transform 2: t1 to mni ---
                transform_t1_to_mni_txt_path = os.path.join(
                    LST_dir, f'sub-{subID}', f'ses-{sesID}', 'temp', f'sub-{subID}_ses-{sesID}_affine_t1w_to_mni.mat')
                transform_t1_to_mni_mat = np.loadtxt(transform_t1_to_mni_txt_path)
                transform_t1_to_mni_obj = ants.create_ants_transform(transform_type='AffineTransform', dimension=3)
                transform_t1_to_mni_obj.set_parameters(list(transform_t1_to_mni_mat[:3,:3].ravel()) + list(transform_t1_to_mni_mat[:3,3]))
                
                transform_t1_to_mni_ants_path = os.path.join(temp_dir, f'sub-{subID}_ses-{sesID}_t1_to_mni_ants.mat')
                ants.write_transform(transform_t1_to_mni_obj, transform_t1_to_mni_ants_path)

                # lesion mask image in MPRAGE space
                print(MPRAGE[i] , mask_image_space_flair , transform_flair_to_mni_ants_path, transform_t1_to_mni_ants_path)
                mask_image = ants.apply_transforms(
                    fixed=ants.image_read(MPRAGE[i]),
                    moving=moving,
                    transformlist=[transform_flair_to_mni_ants_path, transform_t1_to_mni_ants_path],
                    whichtoinvert=[False , True],
                    interpolator='nearestNeighbor'
                )
                # save lesion mask in MPRAGE space temporarily
                mask_image_path = os.path.join(
                    temp_dir, f'sub-{subID}_ses-{sesID}_lesion_mask_in_mprage_space.nii.gz')
                ants.image_write(mask_image, mask_image_path)

                # skip to next case if segmentation already exist
                seg_file = os.path.join(deriv_ses, "inpainting_volumes" , f'sub-{subID}_ses-{sesID}_inpainting_result.nii.gz')
                if os.path.exists(seg_file):
                    print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: lesion inpainting already done, skip and proceed to next case...')
                    continue
                
                run_lit_container(MPRAGE[i], mask_image_path, deriv_ses, dilate)

                inpainting_dir = os.path.join(deriv_ses, "inpainting_volumes")
                raw_result = os.path.join(inpainting_dir, "inpainting_result.nii.gz")
                raw_mask = os.path.join(inpainting_dir, "inpainting_mask.nii.gz")
                raw_masked = os.path.join(inpainting_dir, "inpainting_masked_image.nii.gz")
                raw_original = os.path.join(inpainting_dir, "inpainting_original_image.nii.gz")

                for expected in (raw_result, raw_mask, raw_masked, raw_original):
                    if not os.path.exists(expected):
                        raise FileNotFoundError(
                            f"FS-LIT did not create expected output for sub-{subID}_ses-{sesID}: {expected}"
                        )

                os.rename(raw_result, seg_file)
                os.rename(
                    raw_mask,
                    os.path.join(inpainting_dir, f'sub-{subID}_ses-{sesID}_inpainting_mask.nii.gz'),
                )
                os.rename(
                    raw_masked,
                    os.path.join(inpainting_dir, f'sub-{subID}_ses-{sesID}_inpainting_masked_image.nii.gz'),
                )
                os.rename(
                    raw_original,
                    os.path.join(inpainting_dir, f'sub-{subID}_ses-{sesID}_inpainting_original_image.nii.gz'),
                )

                if os.path.exists(seg_file):
                    print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: Rename FS-LIT lesion inpainted image (BIDS) DONE!')

                # else:
                #     print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: failed to generate FS-LIT, delete ouput folder...')
                #     shutil.rmtree(str(Path(deriv_ses).parent))
                #     if os.path.exists(deriv_ses) or os.path.exists(temp_dir):
                #         raise ValueError(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: failed to delete the derivatives folder(s)!')
                #     else:
                #         print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: successfully deleted the derivatives folder(s)!')
                #         continue

            # except:
            #     print(f'{datetime.datetime.now()} sub-{subID}_ses-{sesID}: Error occured during processing, proceeding with next case.')

#### modify here            
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Run FastSurfer lesion inpainting on cohort.')

    parser.add_argument('-i', '--input_directory', 
                        help='BIDS Folder.', 
                        required=True)

    parser.add_argument('-d', '--derivatives', 
                        help='derivatives folder name in BIDS directory with LST-AI results', 
                        required=True)
    
    parser.add_argument('--dilate', 
                        help='Dilate value for lesion mask.', 
                        type=int, 
                        default=1)
    parser.add_argument('--subjects',
                        help='Comma-separated subject IDs to process, e.g. 001,002,sub-010.',
                        type=str,
                        default=None)
    
    # read the arguments
    args = parser.parse_args()
    
    input_path = str(Path(args.input_directory).resolve())
    

    out_dir = os.path.join(input_path, "derivatives/FS-LIT2")
    if not os.path.exists(out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    # generate list with subject folders for multiprocessing
    data_root = Path(os.path.join(input_path))
    dirs = sorted(list(data_root.glob('*')))
    dirs = [str(x) for x in dirs]
    dirs = [x for x in dirs if "sub-" in x]
    dirs = filter_subject_dirs(dirs, parse_subject_ids(args.subjects))
    print(dirs)
    # check which files have already been processed
    dirs_missing, dirs_processed = availability_check(sub_dirs=dirs,
                                                      deriv_dir=args.derivatives,
                                                      file_suffix='space-FLAIR_label-lesion_mask')
    print(f'Number of incomplete subjects: {len(dirs_missing)}')
    print(f'Number of complete subjects: {len(dirs_processed)}')
    
    # disable multiprocessing
    n_workers = 1

    # only split the list of subjects with missing LST-AI lesion segmentation for multiprocessing
    files = split_list(alist = dirs_missing, 
                       splits = n_workers)

    process_LIT(files[0],
                                               args.derivatives,
                                               out_dir,                                        
                                               args.dilate)

    # # initialize multithreading
    # pool = multiprocessing.Pool(processes=n_workers)
    # # call samseg processing function in multiprocessing setting
    # # for testing: export BIDS='/mnt/d/TWIN_MRI/BIDS_directories/TEST'

    # #./run_lit_containerized.sh --input_image T1w.nii.gz --mask_image lesion_mask.nii.gz --output_directory output_directory --dilate 2

    # for x in range(0, n_workers):
    #     print("starting")
    #     pool.apply_async(process_LIT, args=(files[x],
    #                                            args.derivatives,
    #                                            out_dir,                                        
    #                                            args.dilate))
        
    # pool.close()
    # pool.join()

    print('DONE!')
