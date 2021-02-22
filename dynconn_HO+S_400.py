import os
import glob
import nilearn
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
from nilearn import input_data
import nibabel as nib
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import scipy.io
from nilearn import datasets
import sys


"""FUNCTIONS"""
def extract_fMRI_into_atlas(sub, path, pattern, scan_num, confounds_directory, atlas_dict, output_label, output_directory):
    print("\n\n===Extracting fMRI scans into atlases===")
    # Find all relevant files that match filename "pattern" for subject number "sub". If number of files exceeds "scan_num",
    # select the "scan_num" largest files. Creates a numpy 2D array of shape (number of TRs, number of ROIs)
    directory_path = "%s/sub-%s/func" % (path, sub)
    scan_names = glob.glob(directory_path + "/*Ex*preproc_bold.nii.gz")
    
    if len(scan_names) > scan_num:
        scan_names.sort(key=lambda f: os.stat(f).st_size, reverse=True)
        scan_names = scan_names[:scan_num]

    print("Scan names:\n")
    [print("\t" + scanname) for scanname in scan_names]
    print("-------------------------------------------")
    
    # Check if multiple atlases are provided, if so, extract ROIs and concatenate

    for scan in scan_names:

        # Get corresonding confounds .tsv files for ith scan
        base = "_".join(scan.split("/")[-1].split("_")[:3])
        confounds_filename = confounds_directory + base + '_conformatted.tsv'
        pandas.read_csv(confounds_filename, sep = '\t').fillna(0).to_csv("%s%s_conf_noNA_temp.tsv" % (confounds_directory, base), sep = '\t', index = False)

 
        print("---" + confounds_filename)

        # Load fmri volumes
        func_img = nib.load(scan)
        header_zooms = func_img.header.get_zooms()
        TR = header_zooms[3]
        
        # Initialize Time Series
        time_series = []
        
        # For each atlas, grab what type of masker to use. Extract ROI timeseries using 
        # appropriate masker object and then concatenate all atlas extractions into one 
        # 2D array Timeseries of shape (number of TRs, total number of ROIs across all atlases)
        for atlas_filename, masker in atlas_dict.items():
            print("Extracting into atlas: %s" % atlas_filename)
            # Check which masker to use, then create a masker object and extract scan data into ROI timeseries using masker
            if masker == 'Maps':
                masker = input_data.NiftiMapsMasker(
                    atlas_filename, 
                    resampling_target='data',
                    memory = 'nilearn_cache', 
                    verbose = 0, low_pass=0.1, high_pass=0.01, detrend=True, t_r = TR).fit()
                time_series.append(masker.transform(scan, confounds = '%s%s_conf_noNA_temp.tsv' % (confounds_directory, base)))
            elif masker == "Labels":
                masker = NiftiLabelsMasker(
                    labels_img=atlas_filename,
                    standardize=True,
                    memory='nilearn_cache', 
                    verbose=0, low_pass = 0.1, high_pass = 0.01, detrend = True, t_r = TR)
                time_series.append(masker.fit_transform(scan, confounds = '%s%s_conf_noNA_temp.tsv' % (confounds_directory, base)))

        # Save timeseries to output directory, using output label and base name for reference
        time_series = np.concatenate(time_series, axis = 1)
        print("Saving timeseries to %s/%s_%s_timeseries.npy" % (output_directory, base, output_label))
        np.save("%s/%s_%s_timeseries.npy" % (output_directory, base, output_label), time_series)


def bin_time_series(sub, path, pattern, window_step, window_size):
    print("\n\n===Bin Time Series into Time Windows===")
    print("Binning timeseries into time windows of size %d and step %d" % (window_size, window_step))
    # Get scan filenames from subject id and pattern. Initialize empty containers for storing 
    # the resulting binned time windows and the corresponding start-end indices
    scan_filenames = glob.glob(path + "/sub-%s" % sub + pattern)
    scan_time_windows = {}
    indices = []
    
    # For each time series, create a range of start-end indices based on window step and size parameters. 
    # Bin the time series by the indices, and store the resulting sliding time windows into dicitonary 
    # where each key is a separate scan and its corresponding item is its 3D array of sliding time windows  
    # of shape (time window, TR number, ROI)
    for ts in scan_filenames:

        time_series = np.load(ts)

        time_windows = []
        for i in range(0, time_series.shape[0]-window_size + 1, window_step):
            window = time_series[i:i + window_size, :]
            time_windows.append(window)
            indices.append([i, i+window_size])

        
        scan_time_windows[ts] = np.asarray(time_windows)
    print("Time Series Indices:")
    print(indices)
    return(scan_time_windows)


def connectomes_from_time_windows(time_windows_dict, kind):
    print("\n\n===Compute Connectomes from Time Windows===")
    print("Connectivity Type: %s" % kind)
    # Initialize a connectivity, of type "kind", as well as a dictionary to store the connectomes
    connectivity = ConnectivityMeasure(kind = kind)
    connectomes_dict = {}
    
    # For each scan's 3D array of time windows, compute their connectomes
    for timeseries_name, time_windows in time_windows_dict.items():
        connectomes_dict[timeseries_name] = connectivity.fit_transform(time_windows)
    
    return(connectomes_dict)


def binarize_threshold(connectomes_dict, threshold):
    print("\n\n===Binarize and Threshold Connectomes===")
    # For each 3D Array of Connectomes, go through each Time Window and Binarize it according to 
    # some quantile threshold. Fill diagonal with 0. Return dictionary of 3D Connectome / Scan
    binthresh_connectomes_dict = connectomes_dict.copy()
    
    # Initialize empty translate dict for translating old key names to new key names for MATLAB compatibility
    translate = {}
    scan_num = 0
    
    for timeseries_name, connectomes in binthresh_connectomes_dict.items():
        print("\t%s" % timeseries_name)
        print("\t\t3D Connectome Shape: " + str(connectomes.shape))
	
	# Shorten the dictionary key name to be of a length readable by matlab
        scan_num += 1
        short_timeseries_name = "scan_" + str(scan_num)
        print("\t\tMATLAB key name: " + short_timeseries_name)
        translate[timeseries_name] = short_timeseries_name

	# For each time-windowed connetome, threshold and binarize it
        for i in range(connectomes.shape[0]):
            connectome = connectomes[i].copy()
            np.fill_diagonal(connectome, 0)
            abs_thresh = np.quantile(np.abs(connectome), q = threshold)
            connectome[(connectome>=-abs_thresh) & (connectome<=abs_thresh)] = 0
            connectome[connectome != 0] = 1
            connectomes[i] = connectome

    # Loop through and translate name of dictionary keys to shortened version for MATLAB compatibility
    for old, new in translate.items():
        binthresh_connectomes_dict[new] = binthresh_connectomes_dict.pop(old)
    print("\t\tNew key names: " + str(binthresh_connectomes_dict.keys()))
        
    return binthresh_connectomes_dict


def save_3D_array(binthresh_connectomes_dict, sub, pattern, output_directory, output_label):
    scipy.io.savemat('%s/sub-%s_%s_%s_3Darray.mat' % (output_directory, sub, pattern, output_label), binthresh_connectomes_dict, long_field_names = True)
    print("\n\n---Saved 3D Array to %s/sub-%s_%s_%s_3Darray.mat---" % (output_directory, sub, pattern, output_label))


"""EXECUTION CODE"""


sub = sys.argv[1]

path = "/mnt/chrastil/lab/data/MLINDIV/preprocessed/derivatives/fmriprep"
confounds_directory = "/mnt/chrastil/lab/users/rob/projects/MachineLearning/confounds/"
timeseries_directory = "/mnt/chrastil/lab/users/rob/projects/DynConn/timeseries"
array_directory = "/mnt/chrastil/lab/users/rob/projects/DynConn/3DArrays"

pattern = "Ex"
scan_num = 2
threshold = 0.9


# HO + Schaeffer Atases
output_label = "HOsubcort+Schaefer400_2mm"
ts_pattern = "*Ex*HOsubcort+Schaefer400_2mm*"

schaefer = datasets.fetch_atlas_schaefer_2018(resolution_mm=2, verbose = 0)
schaefer_atlas = schaefer.maps
schaefer_labels = schaefer.labels

ho = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm", verbose = 0)
ho_atlas = ho.maps
ho_labels = ho.labels

atlas_dict = {
    schaefer_atlas: "Labels",
    ho_atlas: "Labels"
}


# Code to execute w/ above parameters
if __name__ == "__main__":
	print("Executing dynconn_HO+S_400.py on subject: " + sub)
	print("===PARAMETERS===\nSubject: %s\nPath: %s\nConfounds Directory: %s\nTimeseries Directory: %s\n3D Array Directory: %s\nScan Pattern: %s\t(Max # of scans used matching pattern: %d)\nBinarize threshold: %f\nOutput Label: %s\nAtlases:\n" % (sub, path, confounds_directory, timeseries_directory, array_directory, pattern, scan_num, threshold, output_label))
	for atlas_file, maskertype in atlas_dict.items():
		print("\t%s\t\tNiftiMasker Used: %s" % (atlas_file, maskertype))
	
	extract_fMRI_into_atlas(sub, path, pattern, scan_num, confounds_directory, atlas_dict, output_label, timeseries_directory)
	time_windows = bin_time_series(sub, timeseries_directory, ts_pattern, window_step = 15, window_size = 45)
	connectomes = connectomes_from_time_windows(time_windows, kind = 'correlation')
	binthresh_connectomes = binarize_threshold(connectomes, threshold)
	save_3D_array(binthresh_connectomes, sub, pattern, array_directory, output_label)

