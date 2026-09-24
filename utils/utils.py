# copied from TWILTGEN/LST_AI_BIDS on 25.04.2025
# modified to suit other analysis methods other than LST_AI

import os
import shutil
from pathlib import Path
import re

# bids helpers
def getSubjectID(path):
    """
    This function extracts the subject ID from the file path (BIDS format)

    Parameters:
    -----------
    path : str
        Path to data file
    
    Returns:
    --------
    found : str 
        BIDS-compliant subject ID
    """
    stringList = str(path).split("/")
    indices = [i for i, s in enumerate(stringList) if 'sub-' in s]
    text = stringList[indices[0]]
    try:
        found = re.search(r'sub-([a-zA-Z0-9]+)', text).group(1)
    except AttributeError:
        found = ''
    return found

def getSessionID(path):
    """
    This function extracts the seesion ID from the file path (BIDS format)

    Parameters:
    -----------
    path : str
        Path to data file
    
    Returns:
    --------
    found : str 
        BIDS-compliant session ID
    """
    stringList = str(path).split("/")
    indices = [i for i, s in enumerate(stringList) if 'ses-' in s]
    text = stringList[indices[0]]
    try:
        found = re.search(r'ses-([a-zA-Z0-9]+)', text).group(1)
    except AttributeError:
        found = ''
    return found

def split_list(alist, splits=1):
    """
    This function splits a list for multiprocessing

    Parameters:
    -----------
    alist : list
        The list that should be split into multiple sub-lists
    
    Returns:
    --------
    alist : list
        This list contains the multiple sub-lists that are used for multiprocessing
    """
    length = len(alist)
    return [alist[i * length // splits: (i + 1) * length // splits]
            for i in range(splits)]

def CopyandCheck(src, dst):
    """
    This function tries to copy a file with original path (src) to a target path (dst) and 
    checks if the src file exists and if it was copied successfully to dst

    Parameters:
    -----------
    src : str
        Full path of original location (with filename in the path)
    dst : str 
        Full path of target location (with filename in the path)

    Returns:
    --------
    None
        The function copies files (with renaming) from an original path (src) to a target path (dst) 
    """
    if os.path.exists(src):
        shutil.copy(src, dst)
        if not os.path.exists(dst):
            raise ValueError(f'failed to copy {src}')
        # else:
        #     print(f'successfully copied {os.path.basename(src)} to {os.path.basename(dst)} target location')
    else:
        raise Warning(f'File {os.path.basename(src)} does not exist in original folder!')

def getfileList(path, suffix):
    """
    This function lists all "*suffix"-files that are in the given path. 

    Parameters:
    -----------
    path : str
        Path to directory in which we want to search for files
    suffix : str
        Suffix of the files we want to list (e.g., "*.nii.gz")
    
    Returns:
    --------
    file_ls : list
        Return the lists of "*suffix"-files.
    """
    file_ls = sorted(list(Path(path).rglob(suffix)))
    return file_ls

def availability_check(sub_dirs, deriv_dir, file_suffix):
    """
    This function checks availability of files with a specific suffix (e.g., "_space-T1w_seg.nii.gz") within a derivatives folder for each subject in the provided list and outputs two lists, 
    one with subjects with missing files and one with subjects for which all files are available. First, the function iterates over all subjects, and for each subject it lists all available 
    sessions and then checks the availability of files for each session in the provided derivatives folder.

    Parameters:
    -----------
    sub_dirs : list
        List with all subject IDs for which availability of files should be checked
    deriv_dir : str
        Path of derivatives folder in which availability of files should be checked
    file_suffix : str
        Suffix of files for which availability should be checked
    
    Returns:
    --------
    sub_missing : list
        List with subjects for which one or more files are missing
    sub_available : list 
        List with subjects for which all files are available
    """
    ### modify here
        # include a mode parameter to check other image types for other methods.
    # initialize empty lists
    sub_missing = []
    sub_available = []

    # iterate through all subject folders
    for sub_dir in sub_dirs:
        # get subject ID
        sub_ID = getSubjectID(sub_dir)

        # list all sessions
        ses_dirs = sorted(list(Path(sub_dir).glob('*')))
        ses_dirs = [str(x) for x in ses_dirs if "ses-" in str(x)]
        
        # initialize availability list 
        any_missing = []

        # iterate through all sessions
        for ses_dir in ses_dirs:
            # get session ID
            ses_ID = getSessionID(ses_dir)

            # check availability of file for this session
            file_path = os.path.join(deriv_dir, f'sub-{sub_ID}', f'ses-{ses_ID}', 'anat', f'sub-{sub_ID}_ses-{ses_ID}_{file_suffix}')
            t1_path = os.path.join(ses_dir, 'anat', f'sub-{sub_ID}_ses-{ses_ID}_T1w.nii.gz')
            flair_path = os.path.join(ses_dir, 'anat', f'sub-{sub_ID}_ses-{ses_ID}_FLAIR.nii.gz')
            if (not os.path.exists(file_path)) and os.path.exists(t1_path) and os.path.exists(flair_path):
                any_missing.append(True)
            else:
                any_missing.append(False)
        
        # check if files are missing for any of the sessions and add subject to sub_missing or sub_available accordingly
        if any(any_missing):
            sub_missing.append(sub_dir)
        else:
            sub_available.append(sub_dir)
    
    return sub_missing, sub_available

def availability_check_systempref(sub_dirs, deriv_dir, file_suffix, system):
    """
    This function checks availability of files with a specific suffix (e.g., "_space-T1w_seg.nii.gz") within a derivatives folder for each subject in the provided list and outputs two lists, 
    one with subjects with missing files and one with subjects for which all files are available. First, the function iterates over all subjects, and for each subject it lists all available 
    sessions and then checks the availability of files for each session in the provided derivatives folder.

    Parameters:
    -----------
    sub_dirs : list
        List with all subject IDs for which availability of files should be checked
    deriv_dir : str
        Path of derivatives folder in which availability of files should be checked
    file_suffix : str
        Suffix of files for which availability should be checked
    
    Returns:
    --------
    sub_missing : list
        List with subjects for which one or more files are missing
    sub_available : list 
        List with subjects for which all files are available
    """
    ### modify here
        # include a mode parameter to check other image types for other methods.

    # Check if system is valid
    if system == 'GH':
        T1w_suffix = 'T1w.nii.gz'
    elif system == 'BMC':
        T1w_suffix = 'T1w.nii.gz'
    else:
        raise ValueError("Invalid system specified. Use 'GH' or 'BMC'.")
    # initialize empty lists
    sub_missing = []
    sub_available = []

    # iterate through all subject folders
    for sub_dir in sub_dirs:
        # get subject ID
        sub_ID = getSubjectID(sub_dir)

        # list all sessions
        ses_dirs = sorted(list(Path(sub_dir).glob('*')))
        ses_dirs = [str(x) for x in ses_dirs if "ses-" in str(x)]
        print(f"Checking subject {sub_ID} with sessions {ses_dirs}")
        
        # initialize availability list 
        any_missing = []

        # iterate through all sessions
        for ses_dir in ses_dirs:
            # get session ID
            ses_ID = getSessionID(ses_dir)

            # check availability of file for this session
            file_path = os.path.join(deriv_dir, f'sub-{sub_ID}', f'ses-{ses_ID}', 'anat', f'sub-{sub_ID}_ses-{ses_ID}_{file_suffix}')
            t1_path = os.path.join(ses_dir, 'anat', f'sub-{sub_ID}_ses-{ses_ID}_{T1w_suffix}')
            flair_path = os.path.join(ses_dir, 'anat', f'sub-{sub_ID}_ses-{ses_ID}_FLAIR.nii.gz')
            if all([(not os.path.exists(file_path)) , os.path.exists(t1_path) , os.path.exists(flair_path)]):
                any_missing.append(True)
            else:
                any_missing.append(False)
            
        
        # check if files are missing for any of the sessions and add subject to sub_missing or sub_available accordingly
        print(f"Any missing? {any(any_missing)}")
        if any(any_missing):
            sub_missing.append(sub_dir)
        else:
            sub_available.append(sub_dir)
    
    return sub_missing, sub_available

def availability_check_fastsurfer(sub_dirs, deriv_dir, file_suffix, system):
    """
    This function checks availability of files with a specific suffix (e.g., "_space-T1w_seg.nii.gz") within a derivatives folder for each subject in the provided list and outputs two lists, 
    one with subjects with missing files and one with subjects for which all files are available. First, the function iterates over all subjects, and for each subject it lists all available 
    sessions and then checks the availability of files for each session in the provided derivatives folder.

    Parameters:
    -----------
    sub_dirs : list
        List with all subject IDs for which availability of files should be checked
    deriv_dir : str
        Path of derivatives folder in which availability of files should be checked
    file_suffix : str
        Suffix of files for which availability should be checked
    
    Returns:
    --------
    sub_missing : list
        List with subjects for which one or more files are missing
    sub_available : list 
        List with subjects for which all files are available
    """
    ### modify here
        # include a mode parameter to check other image types for other methods.

    # Check if system is valid
    if system == 'GH':
        T1w_suffix = 'T1w.nii.gz'
    elif system == 'BMC':
        T1w_suffix = 'T1w.nii.gz'
    else:
        raise ValueError("Invalid system specified. Use 'GH' or 'BMC'.")
    if isinstance(file_suffix, (str, Path)):
        required_files = [str(file_suffix)]
    else:
        required_files = [str(x) for x in file_suffix]

    # initialize empty lists
    sub_missing = []
    sub_available = []

    # iterate through all subject folders
    for sub_dir in sub_dirs:
        # get subject ID
        sub_ID = getSubjectID(sub_dir)

        # list all sessions
        ses_dirs = sorted(list(Path(sub_dir).glob('*')))
        ses_dirs = [str(x) for x in ses_dirs if "ses-" in str(x)]
        # print(f"Checking subject {sub_ID} with sessions {ses_dirs}")
        # initialize availability list 
        any_missing = []

        # iterate through all sessions
        for ses_dir in ses_dirs:
            # get session ID
            ses_ID = getSessionID(ses_dir)

            # check availability of files for this session
            subject_deriv = os.path.join(deriv_dir, f'sub-{sub_ID}', f'ses-{ses_ID}', f'sub-{sub_ID}_ses-{ses_ID}')
            file_paths = []
            for required_file in required_files:
                if os.path.dirname(required_file):
                    file_paths.append(os.path.join(subject_deriv, required_file))
                else:
                    file_paths.append(os.path.join(subject_deriv, 'mri', required_file))

            if "LIT" in deriv_dir:
                t1_path = os.path.join(ses_dir , 'anat', 'inpainting_volumes' , f'sub-{sub_ID}_ses-{ses_ID}_inpainting_result.nii.gz')
            else:
                t1_path = os.path.join(ses_dir, 'anat', f'sub-{sub_ID}_ses-{ses_ID}_{T1w_suffix}')
            # print(f"Checking for file {file_path} and T1w file {t1_path}")
            if any(not os.path.exists(file_path) for file_path in file_paths) and os.path.exists(t1_path):
                any_missing.append(True)
            else:
                any_missing.append(False)
            # print(f"File exists: {os.path.exists(file_path)}, T1w exists: {os.path.exists(t1_path)}")
            # print(any_missing)

        # check if files are missing for any of the sessions and add subject to sub_missing or sub_available accordingly
        if any(any_missing):
            sub_missing.append(sub_dir)
        else:
            sub_available.append(sub_dir)
    
    return sub_missing, sub_available

def return_fsout_list(sub_dirs, deriv_dir, file_suffix="aseg.stats"):
    """
    This function checks availability of FreeSurfer output in BIDS directory.

    Parameters:
    -----------
    sub_dirs : list
        List with all subject IDs for which availability of files should be checked
    deriv_dir : str
        Path of derivatives folder in which availability of files should be checked
    file_suffix : str
        Suffix of files for which availability should be checked
    
    Returns:
    --------
    sub_missing : list
        List with subjects for which one or more files are missing
    sub_available : list 
        List with subjects for which all files are available
    """
    ### modify here
        # include a mode parameter to check other image types for other methods.

    # initialize empty lists
    fs_missing = []
    fs_available = []

    # iterate through all subject folders
    for sub_dir in sub_dirs:
        # get subject ID
        sub_ID = getSubjectID(sub_dir)

        # list all sessions
        ses_dirs = sorted(list(Path(sub_dir).glob('*')))
        ses_dirs = [str(x) for x in ses_dirs if "ses-" in str(x)]
        
        # iterate through all sessions
        for ses_dir in ses_dirs:
            # get session ID
            ses_ID = getSessionID(ses_dir)

            # check availability of file for this session
            file_path = os.path.join(deriv_dir, f'sub-{sub_ID}', f'ses-{ses_ID}',
                                     f'sub-{sub_ID}_ses-{ses_ID}', 'stats' ,f'{file_suffix}')
            
            if not os.path.exists(file_path):
                fs_missing.append(file_path)
            else:
                fs_available.append(file_path)
    
    return fs_missing, fs_available
