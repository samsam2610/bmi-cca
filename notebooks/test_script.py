import sys

sys.modules.values()

from src.utils import *
from src.compare_decoders import *

from src.folder_handler import *
from src.cort_processor import *
from src.cca_processor import *
from src.tdt_support import *
from src.plotter import *
from src.decoders import *
from src.utils import *
from src.filters import *
import pickle
import scipy as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from src.wiener_filter import *
from matplotlib.pyplot import cm
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import os
from itertools import cycle, islice
import copy


def test_func():
    print('test_func')


def produce_subsample_list(start=0.005, step=0.005, big_step=0.05):
    list1 = np.arange(start, .2, step)
    list2 = np.arange(list1[-1] + big_step, 1, big_step)
    subsample_list = [*list1, *list2]
    # subsample_list = np.arange(0.005, 1, .05)
    # print(len(subsample_list))
    print(subsample_list)
    return subsample_list


def process_folders_in_path(path):
    # Split the path into individual folders
    folders = path.split('/')
    
    # Get the last folder (which is "0216----" in this case)
    last_folder = folders[-1]
    
    # Remove any leading and trailing dashes and spaces from the last folder
    cleaned_last_folder = last_folder.strip('- ')
    
    # Join the relevant parts of the path to create the desired string
    name_data = f"rollie-corrected calibration - {cleaned_last_folder}"
    return name_data


def process_path(base_path, saving_path):
    # Use os.listdir to get a list of all items (files and directories) in the base directory
    items = os.listdir(base_path)
    
    # Filter out only the directories from the list of items
    subdirectories = [item for item in items if os.path.isdir(os.path.join(base_path, item))]
    
    # Get the full paths to each subdirectory
    subdirectory_paths = [os.path.join(base_path, subdir) for subdir in subdirectories]
    
    # Print the paths to every folder within the base directory
    for path in subdirectory_paths:
        name_data = process_folders_in_path(path)
        current_session = CortProcessor(path)
        current_session.process()
        current_session.save_processed(save_path=saving_path, save_name=name_data)


path_N9 = '/Users/sam/Dropbox/Tresch Lab/CCA stuffs/rat-fes-data/remove_channels_pickles/n9_removed_channels'
path_N5 = '/Users/sam/Dropbox/Tresch Lab/CCA stuffs/rat-fes-data/remove_channels_pickles/n5_removed_channels'
path_N6 = '/Users/sam/Dropbox/Tresch Lab/CCA stuffs/rat-fes-data/remove_channels_pickles/n6_removed_channels'
path_rollie = '/Users/sam/Dropbox/Tresch Lab/CCA stuffs/rat-fes-data/rollie - corrected calibrations'
path_rollie_pickle = '/Users/sam/Library/CloudStorage/Dropbox/Tresch Lab/CCA stuffs/rat-fes-data/rollie_pickle'
saving_path = '/Users/sam/Dropbox/Tresch Lab/CCA stuffs/rat-fes-data'

subsample_list = produce_subsample_list()

current_cp1_index = 2
current_cp2_index = 3
for current_cp2_index in range(3, 8):
    decoder_rollie = DecodersComparison(cp1_index=current_cp1_index, cp2_index=current_cp2_index, pca_dims=8, subsample_list=subsample_list, path=path_rollie_pickle,
                                        sort_func=lambda x: datetime.strptime(x.split('-')[2], ' %m%d%y'))
    decoder_rollie.reassign_day0_decoder(cp1_list_index=0, cp2_list_index=1)
    decoder_rollie.compare_decoders()
    decoder_rollie.plot_vaf_comparison_multiple(title_str=f'Rollie - VAF for Different Decoders - Multiple - Parallelize - {8} PCA Dims - {current_cp1_index} - {current_cp2_index}', path=saving_path)


# decoder_N9 = DecodersComparison(cp1_index=0, cp2_index=1, subsample_list=subsample_list, path=path_N9)
# process_path(path_rollie, saving_path)
