import tdt
import time
import math
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
from scipy.signal import resample, find_peaks
from src.filters import *
from src.tdt_support import *
from src.neural_analysis import *
from src.wiener_filter import *
from sklearn.model_selection import KFold 
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def process_neural_kinangles(tdt, kin_angles, np_ts, threshold_multiplier,
        crop=(0,0), binsize=0.05, toe_height_path = ''):
    tdt_data = extract_tdt(tdt)
    angles_list, kinematics = extract_anipose_angles(kin_angles)
    if toe_height_path:
        bp_list, bps = extract_anipose_3d(toe_height_path)
        toe_height = bps[0,:,1]
        kinematics = np.vstack((kinematics, toe_height))
        ss = stance_swing_dd(toe_height)
        kinematics = np.vstack((kinematics, ss))
    
    tdt_data, kinematics = crop_data(tdt_data, kinematics, np_ts, crop)

    fs = tdt_data['fs']
    tdt_data['neural'] = filter_neural(tdt_data['neural'], fs) #bandpass
    tdt_data['neural'] = remove_artifacts(tdt_data['neural'], fs)
    
    spikes = autothreshold_crossings(tdt_data['neural'], threshold_multiplier)
    firing_rates = spike_binner(spikes, fs, binsize)

    resampled_angles = resample(kinematics, firing_rates.shape[1], axis=1)
    if toe_height_path:
        resampled_angles[-1,:] = resampled_angles[-1,:] > 0.1
    return firing_rates, resampled_angles

def convert_to_phase(angle):
    peaks, nada = find_peaks(angle, prominence=10)
    peaks = np.append(peaks, np.size(angle))
    peaks = np.insert(peaks, 0, 0)
    gait_phase_list = []

    for i in range(np.size(peaks)-1):
        end = peaks[i+1]
        start = peaks[i]

        phase = np.sin(np.linspace(0.0, 2.0*math.pi, num=end-start,
            endpoint=False))
        gait_phase_list.append(phase)
    
    return np.hstack(gait_phase_list), gait_phase_list


def convert_to_limbfootphase(limbfoot_angle):
    peaks = extract_peaks(limbfoot_angle, 115)
    gait_phase_list = []
    for i in range(np.size(peaks)-1):
        end = peaks[i+1]
        start = peaks[i]

        phase = np.sin(np.linspace(0.0, 2.0*math.pi, num=end-start,
            endpoint=False))
        gait_phase_list.append(phase)

    starting_index = peaks[0]
    ending_index = peaks[-1]
    return np.hstack(gait_phase_list), starting_index, ending_index, peaks

def stance_swing(toe_height):
    peaks = extract_peaks(toe_height, 12)
    peaks = np.append(peaks, np.size(toe_height))
    peaks = np.insert(peaks, 0, 0)
    ss_list = []
    

    for i in range(np.size(peaks)-1):
        
        end = peaks[i+1]
        start = peaks[i]
        

        gait = toe_height[start:end]
        #dx = np.gradient(gait)
        #ddx = np.gradient(dx)

        #gait_peaks = extract_peaks(ddx, 0.02)
        minny = np.amin(gait)
        ss = gait > minny + 2

        ss_list.append(ss)


    stance_swing = np.hstack(ss_list)

    return stance_swing

def stance_swing_forelimb(forelimb):
    peaks, nada = find_peaks(forelimb, prominence=10)
    peaks = np.append(peaks, np.size(forelimb))
    peaks = np.insert(peaks, 0, 0)
    ss_list = []
    for i in range(np.size(peaks)-1):
        end = peaks[i+1]
        start = peaks[i]

        temp = forelimb[start:end]
        valley = start + np.argmin(temp)

        stance = np.ones(forelimb[start:valley].shape)
        swing = np.zeros(forelimb[valley:end].shape)
        ss = np.hstack((stance, swing))
        ss_list.append(ss)

    stance_swing = np.hstack(ss_list)
    return stance_swing

def stance_swing_dd(toe_height):
    peaks = extract_peaks(toe_height, 12)
    peaks = np.append(peaks, np.size(toe_height))
    peaks = np.insert(peaks, 0, 0)
    ss_list = []
    print(toe_height.shape)

    for i in range(np.size(peaks)-1):
        end=peaks[i+1]
        start=peaks[i]

        gait = toe_height[start:end]
        dx = np.gradient(gait)
        ddx = np.gradient(dx)

        ddx_peaks = extract_peaks(ddx, 0.02)
        if np.size(ddx_peaks) == 2:
            ss = np.ones(np.size(gait), dtype=bool)
            ss[ddx_peaks[0]:ddx_peaks[1]] = 0
        else:
            minny = np.amin(gait)
            ss = gait>minny+2

        ss_list.append(ss)

    stance_swing = np.hstack(ss_list)
    return stance_swing


    




def linear_decoder(firing_rates, kinematics, n=10, l2=0):
    #rates_format, angles_format = format_data(firing_rates.T, kinematics.T, n)
    h = train_wiener_filter(rates_format, angles_format, l2)

    return h

def decode_kfolds(rates, kins, k=10):
    kf = KFold(n_splits=k)

    h_list = []

    vaf_array = np.zeros((kins.shape[1], k))
    index=0
    best_vaf=0
    for train_index, test_index in kf.split(rates):


        train_x, test_x = rates[train_index, :], rates[test_index,:]
        train_y, test_y = kins[train_index, :], kins[test_index, :]

        h=train_wiener_filter(train_x, train_y)
        predic_y = test_wiener_filter(test_x, h)
        
        for j in range(predic_y.shape[1]):
            vaf_array[j, index] = vaf(test_y[:,j], predic_y[:,j])
            
        if vaf_array[3, index] > best_vaf:
            best_vaf = vaf_array[3, index]
            best_h = h
            final_test_x = test_x
            final_test_y = test_y

        index = index+1
    
    return best_h, vaf_array, final_test_x, final_test_y


def classify_kfolds(rates, stance_swing, k=10):
    kf = KFold(n_splits=k)

    accuracy_list = []
    best_accuracy=0
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=10000,
       tol=1e-3))
    best_model = clf
    for train_index, test_index in kf.split(rates):


        train_x, test_x = rates[train_index, :], rates[test_index,:]
        train_y, test_y = stance_swing[train_index], stance_swing[test_index]

        clf.fit(train_x, train_y)
        y_pred = clf.predict(test_x)
        pred_accuracy = accuracy_score(test_y, y_pred)
        accuracy_list.append(pred_accuracy)
        if best_accuracy < pred_accuracy:
            best_accuracy = pred_accuracy
            best_model = clf
            final_test_x = test_x
            final_test_y = test_y
    
    return best_model, accuracy_list, final_test_x, final_test_y

def decode_kfolds_forelimb(forelimb_kins, hindlimb_kins, k=10):
    kf = KFold(n_splits=k)

    h_list = []

    vaf_array = np.zeros((hindlimb_kins.shape[1], k))
    index=0
    best_vaf=0
    for train_index, test_index in kf.split(forelimb_kins):


        train_x, test_x = forelimb_kins[train_index], forelimb_kins[test_index]
        train_y, test_y = hindlimb_kins[train_index, :], hindlimb_kins[test_index, :]

        h=train_wiener_filter(train_x, train_y)
        predic_y = test_wiener_filter(test_x, h)
        h_list.append(h)
        for j in range(predic_y.shape[1]):
            vaf_array[j, index] = vaf(test_y[:,j], predic_y[:,j])
         
        if vaf_array[2, index] > best_vaf:
            best_vaf = vaf_array[2, index]
            best_h = h
            final_test_x = test_x
            final_test_y = test_y


        index=index+1
    
    return best_h, vaf_array, final_test_x, final_test_y



def stitch_data(rates_list, kin_list, n=10):
    formatted_rates = []
    formatted_angles = []

    for i in range(len(rates_list)):
        f_rate, f_angle = format_data(rates_list[i].T, kin_list[i].T, n)
        formatted_rates.append(f_rate.T)
        formatted_angles.append(f_angle.T)

    rates = np.hstack(formatted_rates).T
    kin = np.hstack(formatted_angles).T
    return rates, kin

def extract_peaks(angles, thres_height=115):
    peaks, nada = find_peaks(angles, height=thres_height)
    return peaks

def find_bad_gaits(peaks):
    average_gait_samples = np.average(np.diff(peaks))
    above = 1.25 * average_gait_samples
    below = .8 * average_gait_samples

    bad_above = np.argwhere(np.diff(peaks) > above)
    bad_below = np.argwhere(np.diff(peaks) < below)

    bads = np.squeeze(np.concatenate((bad_above, bad_below)))

    return bads.tolist()

def remove_bad_gaits(rates, angles, peak_threshold=115):
    peaks = extract_peaks(angles[3,:], peak_threshold)
    bads = find_bad_gaits(peaks)

    rates_list = []
    angles_list = []

    for i in range(np.size(peaks)-1):
        if i in bads:
            continue
        first = peaks[i]
        last=peaks[i+1]
        gait_rates = rates[:, first:last]
        gait_angles = angles[:, first:last]

        rates_list.append(gait_rates)
        angles_list.append(gait_angles)


    rebuilt_rates = np.hstack(rates_list)
    rebuilt_angles = np.hstack(angles_list)
    return rebuilt_rates, rebuilt_angles 

def extract_gait(rates, angles, thres, bool_resample=False):
    peaks = extract_peaks(angles,thres)
    rates_list = []
    angles_list = []

    avg_samples = int(np.round(np.average(np.diff(peaks))))

    for i in range(np.size(peaks)-1):
        first = peaks[i]
        last = peaks[i+1]
        gait_rates = rates[:, first:last]
        gait_angles = angles[first:last]
        if bool_resample:
            gait_rates = resample(gait_rates, avg_samples, axis=1)
            gait_angles = resample(gait_angles, avg_samples)
        rates_list.append(gait_rates)
        angles_list.append(gait_angles)

    return rates_list, angles_list, peaks

def vid_from_gait(crop, angles_list, gait_number, video, peaks, filename, binsize=0.05,
        framerate=1):
    bin_number = np.size(np.hstack(angles_list[0:gait_number]))

    frame_iter = int(binsize*200)
    start_frame = crop[0]*200 + ((peaks[0]+bin_number) * frame_iter)
    end_frame = start_frame + len(angles_list[gait_number]) * frame_iter

    directory = '/home/diya/Documents/rat-fes/results/movies/{}-{}/'.format(filename, gait_number)
    os.mkdir(directory)
    
    angles = angles_list[gait_number]

    fig0 = plt.figure(figsize = (720/96, 440/96), dpi=96)
    ax0 = fig0.add_subplot(111)
    ax0.set_xlim(0, np.size(angles)-1)
    ax0.set_ylim(np.min(angles), np.max(angles))
    
    img_list = []

    for i in range(np.size(angles)):
        y = angles[0:i+1]
        x = np.arange(0,i+1)
        degree_num = int(angles[i])
        ax0.plot(x, y, c='blue')
        plt.savefig(directory + 'degree{}_{}'.format(i, degree_num), )
        time.sleep(.1)
        img_list.append(cv2.imread(directory +
            'degree{}_{}.png'.format(i, degree_num)))

    height = img_list[0].shape[0]
    width = img_list[0].shape[1]
    out = cv2.VideoWriter(directory + 'kin.mp4',
            cv2.VideoWriter_fourcc(*'DIVX'), framerate,(width, height))

    for img in img_list:
        out.write(np.array(img))

    out.release()
    

    cap = cv2.VideoCapture(video)
    img_list = []
    
    j=0
    for k in range(start_frame, end_frame, frame_iter):
        cap.set(1, k)
        ret, frame = cap.read()
        cv2.imwrite(directory + 'live_frame{}_{}.png'.format(k, int(angles[j])), frame)
        img_list.append(frame)
        j=j+1
    
    height = img_list[0].shape[0]
    width = img_list[0].shape[1]

    out = cv2.VideoWriter(directory + 'live_video.mp4',
            cv2.VideoWriter_fourcc(*'DIVX'), framerate,(width, height))
    for img in img_list:
        out.write(np.array(img))

    out.release()



