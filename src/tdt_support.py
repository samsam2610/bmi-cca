import tdt
import numpy as np
import pandas as pd
import copy

def deprec_extract_tdt(tdt_file):
    tdt_dict = {}

    data = tdt.read_block(tdt_file)
    tdt_dict['neural'] = data.streams.Wav1.data*1000000 #in microvolts
    tdt_dict['fs'] = data.streams.Wav1.fs
    tdt_dict['ts'] = np.arange(0, tdt_dict['neural'].shape[1] / tdt_dict['fs'], 
            1/tdt_dict['fs'])
    tdt_dict['pulse_time'] = data.epocs.U11_.onset[0]
    
    return tdt_dict

def extract_tdt(tdt_file, npts_file):
    tdt_dict = {}

    data = tdt.read_block(tdt_file)
    tdt_dict['neural'] = (data.streams.Wav1.data*1000000).T #in microvolts
    tdt_dict['fs'] = data.streams.Wav1.fs
    tdt_dict['ts'] = np.arange(0, tdt_dict['neural'].shape[0] / tdt_dict['fs'], 
            1/tdt_dict['fs'])
    tdt_dict['pulse_time'] = data.epocs.Out1.onset[0]
    
    tdt_dict['cam_timestamps'] = np.load(npts_file)

    return tdt_dict



def extract_kin_data(coords_csv, angles_csv):
    kin_data = {}
    bp_list, coords = extract_anipose_3d(coords_csv)
    angles_list, angles = extract_anipose_angles(angles_csv)
    kin_data['bodyparts'] = bp_list
    kin_data['coords'] = coords
    kin_data['angles_list'] = angles_list
    kin_data['angles'] = angles

    return kin_data

def get_sync_sample(tdt_data):
    cam_ts = tdt_data['cam_timestamps']
    delay = cam_ts[1]-cam_ts[0]
    start_time = tdt_data['pulse_time'] + delay
    sample_number = start_time * tdt_data['fs']
    return round(sample_number)

def deprec_get_sync_sample(np_ts, tdt_data):
    cam_ts = np.load(np_ts)
    delay = cam_ts[1]-cam_ts[0]
    start_time = tdt_data['pulse_time'] + delay
    sample_number = start_time * tdt_data['fs']
    return round(sample_number)

def extract_anipose_3d(csv):
    df = pd.read_csv(csv)
    long_bp_list = df.columns
    bp_list = []
    for bodypart in long_bp_list:
        temp = bodypart.split('_')[0]
        stop_here = 'M'
        if temp is stop_here:
            break
        if temp not in bp_list:
            bp_list.append(temp)
    
    ret_arr = np.empty((df.index.size, len(bp_list), 3))
    
    for idx, bodypart in enumerate(bp_list):
        ret_arr[:, idx, 0] = df[bodypart + '_x']
        ret_arr[:, idx, 1] = df[bodypart + '_y']
        ret_arr[:, idx, 2] = df[bodypart + '_z']
    #if filtering==True:
    #    clear_list = df.loc[df['ScaRot_ncams'].isnull()].index.tolist()
    #    for element in clear_list:
    #        ret_arr[:, element, :] = np.nan
    
    return bp_list, ret_arr
            
def extract_anipose_angles(csv):
    df = pd.read_csv(csv)
    df = df.iloc[:,0:df.columns.get_loc('fnum')]
    bp_list = df.columns.to_list()
    angles_list = []
    for column in df:
        angles_list.append(df[column].to_numpy())

    return bp_list, np.array(angles_list).T


def deprec_crop_data(tdt_data, kinematics, np_ts, crop=(0,70)):
    #add in start_time/end_time in video, output cropped cortical/kin data
    start_time = crop[0]
    end_time = crop[1]

    kin_start = start_time*200
    kin_end = end_time*200

    init_start_sample = get_sync_sample(np_ts, tdt_data)

    start_sample = round(init_start_sample + (start_time * tdt_data['fs']))
    end_sample = round(init_start_sample + (end_time * tdt_data['fs']))


    temp_neural = tdt_data['neural'] #going to slice variable
    temp_ts = tdt_data['ts']
    tdt_data['neural'] = temp_neural[:,start_sample:end_sample]
    tdt_data['ts'] = temp_ts[start_sample:end_sample]

    if kinematics.ndim==3:
        kinematics = kinematics[:, kin_start:kin_end,:]
    elif kinematics.ndim==2:
        kinematics = kinematics[:, kin_start:kin_end]
    elif kinematics.ndim==1:
        kinematics = kinematics[kin_start:kin_end]

    return tdt_data, kinematics

def crop_data_tdt(tdt_file, np_ts, start_time, end_time):

    tdt_data = extract_tdt(tdt_file)

    init_start_sample = get_sync_sample(np_ts, tdt_data)

    start_sample = round(init_start_sample + (start_time * tdt_data['fs']))
    end_sample = round(init_start_sample + (end_time * tdt_data['fs']))

    temp_neural = tdt_data['neural'] #going to slice variable
    tdt_data['neural'] = temp_neural[:,start_sample:end_sample]


    return tdt_data



