from src.folder_handler import *
from src.tdt_support import *
import math
import pickle
import scipy as spicy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from src.wiener_filter import *
from matplotlib.pyplot import cm
from scipy import signal
import scipy.stats as stats


def arctan_fn(predicted_sin, predicted_cos):
    arctans = []
    for i in range(predicted_sin.shape[1]):
        arctan = np.arctan2(predicted_sin[:, i], predicted_cos[:, i])
        arctan_angles = np.degrees(arctan) + 180
        arctans.append(arctan_angles)
    arctans = np.array(arctans).T
    return arctans


def sine_and_cosine(phase_list):
    phase_list = np.radians(phase_list - 180)
    sin_array = []
    cos_array = []
    for i in range(phase_list.shape[1]):
        sin = np.sin(phase_list[:, i])
        cos = np.cos(phase_list[:, i])
        sin_array.append(sin)
        cos_array.append(cos)
    sin_array = np.array(sin_array).T
    cos_array = np.array(cos_array).T
    return sin_array, cos_array


def predicted_lines(actual, H):
    holding_array = []
    for ii in range(H.shape[1]):
        temp1 = test_wiener_filter(actual, H[:, ii])
        holding_array.append(temp1)
    holding_array = np.array(holding_array).T
    return holding_array


def to_phasex(peaks, angles):
    for i in range(0, peaks.shape[0] - 1):
        for j in range(0, peaks[i + 1] - peaks[i]):
            angles[peaks[i] + j] = j * 360 / (peaks[i + 1] - peaks[i])
        angles[-1] = 0
    return angles


def tailored_peaks(angles, index, name):
    peak_dict = {'foot': {'signal': -(angles[:, index]), 'window': 60, 'prominence': 10, 'distance': 6,
        'height': -np.mean(angles[:, index])},
        'knee': {'signal': angles[:, index], 'window': 60, 'prominence': 10, 'distance': None,
            'height': np.mean(angles[:, index])},
        'hip': {'signal': angles[:, index], 'window': 60, 'prominence': 3, 'distance': 5, 'height': None},
        'limbfoot': {'signal': angles[:, index], 'window': 60, 'prominence': 10, 'distance': None,
            'height': np.mean(angles[:, index])},
        'hand': {'signal': -(angles[:, index]), 'window': 60, 'prominence': 10, 'distance': 6,
            'height': -np.mean(angles[:, index])},
        'elbow': {'signal': angles[:, index], 'window': 60, 'prominence': 6.5, 'distance': 5, 'height': None},
        'shoulder': {'signal': -(angles[:, index]), 'window': 60, 'prominence': 5, 'distance': None,
            'height': -1.1 * np.mean(angles[:, index])},
        'forelimb': {'signal': angles[:, index], 'window': 60, 'prominence': 9, 'distance': 5, 'height': None},
        'ankle': {'signal': -(angles[:, index]), 'window': 60, 'prominence': 10, 'distance': 6,
            'height': -np.mean(angles[:, index])}}
    peaks, _ = spicy.signal.find_peaks(peak_dict[name]['signal'], wlen=peak_dict[name]['window'],
                                       prominence=peak_dict[name]['prominence'], distance=peak_dict[name]['distance'],
                                       height=peak_dict[name]['height'])
    peaks = np.concatenate([[0], peaks, [np.shape(angles[:, index])[0] - 1]])
    return peaks


def align_to_hind(phases, target):
    temp_shift = []
    ts = np.linspace(0, (phases.shape[0] * 50) / 1000, phases.shape[0])
    dx = np.mean(np.diff(ts))
    for i in range(phases.shape[1]):
        if i != target:
            shift = (np.argmax(signal.correlate(phases[:, i], phases[:, target])) - phases[:, target].shape[0]) * dx
            shift = int(-shift * 1000 / 50)
            temp_shift = np.append(temp_shift, shift)
        else:
            temp_shift = np.append(temp_shift, 0)
    return temp_shift


def best_hindlimb_match(phase_list, roll, AOI):
    rank_list = []
    for i in range(4, 7):
        mate = np.roll(phase_list[:, i], int(roll[i]))
        target_phase = phase_list[:, AOI]
        r, p = stats.pearsonr(mate, target_phase)
        # phase_synchrony = 1-np.sin(np.abs(target_phase-mate)/2)
        # rank = np.mean(phase_synchrony)
        # rank_list.append(rank)
        rank_list.append(r)
    best_index = max(range(len(rank_list)), key=rank_list.__getitem__) + 4
    return best_index


def impulse_response(AOI, H_mat, phase_list, plotting=False):
    column_response = []
    for i in range(0, 32):
        product_list = []
        for j in range(0, 10):
            dummyarray = np.zeros((10, 32))
            dummyarray[-1, i] = 1
            dummyarray = np.roll(dummyarray, -j, axis=0)
            dummyarray = dummyarray.flatten()
            dummyarray = np.insert(dummyarray, 0, 1)
            dummyarray = dummyarray.reshape(321, 1)
            dummyarray = dummyarray.T
            product = np.dot(dummyarray, H_mat)
            product_list.append(product[0])
        product_list = np.array(product_list)
        column_response.append(product_list)
    av_gait_bins = average_gait_bins(phase_list, AOI)
    padded_column_response = []
    if (av_gait_bins - 10) >= 1:
        n = av_gait_bins - 10
        for i in range(len(column_response)):
            padded_column = np.append(column_response[i][:, AOI], np.full((n, 1), np.mean(column_response[i][:, AOI])))
            padded_column_response.append(padded_column)
        padded_column_response = np.array(padded_column_response)
    if plotting == True:
        fig1, ax1 = plt.subplots()
        x = np.arange(0, 10, 1)
        for i in range(len(column_response)):
            ax1.plot(x, column_response[i][:, AOI])
        ax1.set_xticks(x)
    return padded_column_response


def rates_by_gait(full_phase_list, rates_array, AOI, count_thresh, plotting=False):
    phase_list = full_phase_list[:, AOI]
    unique_val, unique_first_index, unique_count = np.unique(phase_list, return_index=True, return_counts=True)
    unique_range = len(unique_count)
    rates_by_phase = []
    phase_tracker = []
    for i in range(1, unique_range):
        if unique_count[i] >= count_thresh:
            phase_tracker = np.append(phase_tracker, unique_val[i])
            hodl = np.where(phase_list == unique_val[i])
            hold = hodl[0]
            hold_rates = []
            for i in range(len(hold)):
                rates_row = rates_array[hold[i], -32:]
                hold_rates.append(rates_row)
            hold_rates = np.mean(hold_rates, axis=0)
            rates_by_phase.append(hold_rates)
    rates_by_phase = np.array(rates_by_phase)
    if plotting == True:
        fig1, ax1 = plt.subplots()
        x = phase_tracker
        for i in range(rates_by_phase.shape[1]):
            ax1.plot(x, rates_by_phase[:, i], alpha=0.5)
    return rates_by_phase, phase_tracker


def average_gait_bins(phase_list, AOI):
    gait_lengths = []
    gait_indicies = np.where(phase_list[:, AOI] == 0)[0]
    for i in range(1, gait_indicies.shape[0]):
        gait_length = gait_indicies[i] - gait_indicies[i - 1]
        gait_lengths = np.append(gait_lengths, gait_length)
    av_gait_length = math.ceil(np.average(gait_lengths))
    return av_gait_length
