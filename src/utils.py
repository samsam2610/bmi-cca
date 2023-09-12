import tdt
import time
import math
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import cm
import cv2
import pickle
from src.folder_handler import *
from src.cort_processor import *
from src.cca_processor import *
from src.tdt_support import *
from src.plotter import *
from src.decoders import *
from src.utils import *
from src.filters import *
from src.wiener_filter import *
import scipy as sio
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from itertools import cycle, islice
import copy
import sys


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def classif_accuracy(array1, array2):
    return np.divide(np.sum(array1 == array2), np.size(array1))


def flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)


def bins_to_seconds(samples, binsize=50):
    return np.linspace(0, (samples * binsize) / 1000, samples)


def load_data_from_folder(path, exporting_graph=False, raw_data=False):
    """
    Loads data from files in the specified folder path and returns a dictionary of data objects.

    Parameters:
    path (str): The path to the folder containing the data files.
    exporting_graph (bool, optional): Whether to export a graph. Defaults to False.
    raw_data (bool, optional): Whether to load raw data. Defaults to False.

    Returns:
    data_dict: contains all the cort_processor objects
    sorted_keys: list of keys for the above dict sorted by dates

    Examples:
    path = '/Users/sam/Dropbox/Tresch Lab/CCA stuffs/rat-fes-data/remove_channels_pickles/n9_removed_channels'
    data_dict, sorted_keys = load_data_from_folder(path=path)
    """
    filenames = os.listdir(path)
    
    temp_datasets = []
    temp_var_names = []
    data_dict = {}
    file_list = []
    plt.close("all")
    ignore_extensions = ('.pdf', '.DS_Store')
    for file in filenames:
        if file.endswith(ignore_extensions):
            continue  # ignore this file
        file_list.append(os.path.splitext(file)[0])
        
        current_path = path + '/' + file
        
        if raw_data:
            cp = CortProcessor(current_path)  # Load mat file from Filippe data
        else:
            with open(current_path, 'rb') as inp:
                cp = pickle.load(inp)
        
        # Add data object to dict with respected name
        dict_name = '_'.join(os.path.splitext(file)[0].split('_')[:2])
        data_dict[dict_name] = cp
        
        if exporting_graph:
            h, vaffy, test_x, test_y = cp.decode_angles()
            predic = test_wiener_filter(test_x, h)
            samples = test_x.shape[0]
            
            # Font data for exporting figure
            plt.rcParams['pdf.fonttype'] = 42
            plt.rcParams['ps.fonttype'] = 42
            
            ts = np.linspace(0, (samples * 50) / 1000, samples)
            
            fig, ax = plt.subplots()
            fig.set_size_inches(10, 2)
            ax.plot(ts, test_y[:, 1], color='black')
            ax.plot(ts, predic[:, 1], color='red')
            ax.set_xlim((10, 15))
            ax.axis('off')
            print(vaf(predic[:, 1], test_y[:, 1]))
            
            current_pdf_file = current_path.replace(".mat", ".pdf")
    
    from datetime import datetime
    
    keys = list(data_dict.keys())  # get a list of all keys in the dictionary
    sorted_keys = sorted(keys, key=lambda x: datetime.strptime(x.split('_')[1], '%y%m%d'))
    print(sorted_keys)
    
    return data_dict, sorted_keys


def describe_dict(input_dict):
    """
    Prints information about the keys and arrays in the input dictionary.

    Parameters:
    input_dict (dict): A dictionary containing keys and corresponding arrays.

    Returns:
    None
    """
    keys_list = list(input_dict.keys())
    # Print the list of keys
    print(f"List of keys {keys_list}")
    
    # Loop through each key in the dictionary
    for key in input_dict:
        if key != 'angle_names' and key != 'which_channels':
            # Loop through each array in the list value associated with the current key
            for array in input_dict[key]:
                # Get the shape (dimensions) of the current array
                array_size = len(array)
                array_shape = len(array[0])
                
                # Print the key, index of the current array within the list, and its dimensions
                print(f"Key '{key}' has length {array_size} with {array_shape} columns")


def split_range_into_groups(list_of_numbers, num_groups, step=1):
    groups = {}
    groups_index = {}
    list_length = len(list_of_numbers)
    if num_groups > list_length:
        num_groups = list_length
    
    for i in np.arange(0, num_groups, 1):
        groups[i], groups_index[i] = [], []
        currentIndex = i + 1
        startIndex = 1
        # Populate the current group
        while currentIndex <= len(list_of_numbers):
            groups[i].append(list_of_numbers[currentIndex - 1])
            groups_index[i].append(currentIndex - 1)
            currentIndex = i + 1 + num_groups * startIndex
            startIndex += 1
        
        # # Sort the group backward if i is odd  # if i % 2 == 1:  #     groups[i] = groups[i][::-1]  #     groups_index[i] = groups_index[i][::-1]
    
    return groups, groups_index


def plot_vaf_comparison_multiple(number_of_gaits, subsample_list, title_str, path, max_gait=65, min_gait=0, ylim=None,
                                 fig_size=(15, 5), fixed_decoder_scores=None, pca_decoder_scores=None,
                                 cca_decoder_scores=None, r_scores=None, pinv_scores=None):
    """
    Plots the VAF (Variance Accounted For) comparison for different decoders.

    Parameters:
    number_of_gaits (int): The number of gaits to plot.
    subsample_list (list): The list of subsample percentage
    title_str (str): The title string for the plot.
    path (str): The path to save the plot.
    max_gait (int, optional): The maximum gait number to include. Defaults to 65.
    ylim (tuple, optional): The limits for the y-axis as a tuple (bottom, top). Defaults to None.
    fig_size (tuple, optional): The size of the figure (in inches, width heigh). Defaults to [15, 5]
    fixed_decoder_scores (list, optional): The scores of the fixed decoder. Defaults to None.
    pca_decoder_scores (list, optional): The scores of the PCA decoder. Defaults to None.
    cca_decoder_scores (list, optional): The scores of the CCA decoder. Defaults to None.
    r_scores (list, optional): The scores of the R decoder. Defaults to None.
    pinv_scores (list, optional): The scores of the pinv decoder. Defaults to None.

    Returns:
    None
    """
    
    plt.close("all")
    # Font data for exporting figure
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    fig = plt.figure()
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax1 = fig.add_subplot(111)
    x = subsample_list
    if ylim is not None:
        ax1.set_ylim(bottom=ylim[0], top=ylim[1])
    else:
        ax1.set_ylim(bottom=-1, top=1)
    
    ax1.set_yticks(np.arange(-1, 1.2, 0.2))  # Set y-axis tick locations
    
    # calculate the mean and std of each of the decoder's score
    num_gaits = []
    pca_scores_avg, pca_scores_std = [], []
    cca_scores_avg, cca_scores_std = [], []
    r_scores_avg, r_scores_std = [], []
    pinv_scores_avg, pinv_scores_std = [], []
    fixed_decoder_score_clean = []
    for i in range(len(subsample_list)):
        num_gaits.append(int(np.average(number_of_gaits[i])))
        if pca_decoder_scores is not None:
            pca_scores_avg.append(np.average(pca_decoder_scores[i]))
            pca_scores_std.append(np.std(pca_decoder_scores[i]))
        
        if cca_decoder_scores is not None:
            cca_scores_avg.append(np.average(cca_decoder_scores[i]))
            cca_scores_std.append(np.std(cca_decoder_scores[i]))
        
        if r_scores is not None:
            r_scores_avg.append(np.average(r_scores[i]))
            r_scores_std.append(np.std(r_scores[i]))
        
        if pinv_scores is not None:
            pinv_scores_avg.append(np.average(pinv_scores[i]))
            pinv_scores_std.append(np.std(pinv_scores[i]))
        
        if fixed_decoder_scores is not None:
            if isinstance(fixed_decoder_scores, dict):
                fixed_decoder_score_clean.append(fixed_decoder_scores[i][0])
            else:
                fixed_decoder_score_clean = fixed_decoder_scores
    
    if fixed_decoder_scores is not None:
        ax1.plot(num_gaits, fixed_decoder_score_clean, '--', color='tab:pink', label='fixed_decoder')
    if pca_decoder_scores is not None:
        peak = np.full(len(num_gaits), max(pca_scores_avg))
        ax1.errorbar(num_gaits, pca_scores_avg, yerr=pca_scores_std, marker='o', color='tab:blue', label='pca decoder')
        ax1.plot(num_gaits, peak, '--', color='tab:blue', label='max pca decoder')
    if cca_decoder_scores is not None:
        ax1.errorbar(num_gaits, cca_scores_avg, yerr=cca_scores_std, marker='s', color='tab:green', label='cca aligned')
    if r_scores is not None:
        ax1.errorbar(num_gaits, r_scores_avg, yerr=r_scores_std, marker='*', color='tab:orange',
                     label='preloaded decoder')
    if pinv_scores is not None:
        ax1.errorbar(num_gaits, pinv_scores_avg, yerr=pinv_scores_std, marker='+', color='tab:purple',
                     label='pinv decoder')
    
    ax1.legend(loc='lower right')
    
    if max_gait > max(number_of_gaits):
        max_gait = max(number_of_gaits)
    
    ax1.set_xlim(min_gait, max_gait)
    ax1.set_xlabel('Number of Gait Cycles Trained On')
    ax1.set_ylabel('VAF')
    plt.title(title_str)
    plt.show()
    
    pdf_file_path = os.path.join(path, title_str + '.pdf')
    plt.savefig(pdf_file_path)


def test_data():
    print('hello world')



