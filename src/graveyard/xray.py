import pandas as pd
import numpy as np

def extract_frames(df, iterate=3):
    df = df.drop('Time', axis=1)
    bp_list=[]
    temp_columns_list = df.columns.tolist()
    temp_string = 'Bahn'
    columns_list = []
    for element in temp_columns_list:
        if temp_string not in element:
            columns_list.append(element)
    for i in range(0, len(columns_list), iterate):
        bp = columns_list[i].split('X')[0].strip()
        if not pd.isnull(df[bp+' X']).all():
            bp_list.append(bp)
    if iterate==2:
        dims=2
    else:
        dims=3
    coords = np.zeros((len(bp_list), len(df.index), 3))
    for j in range(len(bp_list)):
        bp = bp_list[j]
        coords[j, :, 0] = np.array(df[bp+' X'])
        coords[j,:,1] = np.array(df[bp+' Y'])
        if iterate>2:
            coords[j,:,2] = np.array(df[bp+' Z'])

    return bp_list, coords

    
def triangulate_dlc(arr_a, arr_b, mtx_a, dist_a, mtx_b, dist_b, p1, p2):
    
    ret_array = np.empty((arr_a.shape[0], arr_a.shape[1], 3))
    ret_array[:] = np.nan
    
    for i in range(arr_a.shape[0]):
        out_a = np.squeeze(cv2.undistortPoints(arr_a[i,:,:], mtx_a, dist_a))
        out_b = np.squeeze(cv2.undistortPoints(arr_b[i,:,:], mtx_b, dist_b))
        ret_array[i,:,:] = cv2.triangulatePoints(p1, p2, out_a.T, out_b.T)[:3, :].T
    return ret_array

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
    
    ret_arr = np.empty((len(bp_list), df.index.size, 3))
    
    for idx, bodypart in enumerate(bp_list):
        ret_arr[idx, :, 0] = df[bodypart + '_x']
        ret_arr[idx, :, 1] = df[bodypart + '_y']
        ret_arr[idx, :, 2] = df[bodypart + '_z']
    #if filtering==True:
    #    clear_list = df.loc[df['ScaRot_ncams'].isnull()].index.tolist()
    #    for element in clear_list:
    #        ret_arr[:, element, :] = np.nan
    
    return bp_list, ret_arr
            
        
def extract_dlc_frames(df):
    bp_list = list(dict.fromkeys(df.iloc[0]))
    df.drop(0, inplace=True)
    coords = np.zeros((len(bp_list), len(df.index), 2))
    for i in range(len(bp_list)):
        if i>0:
            str_i = '.' + str(i)
        else:
            str_i = ''
        for j in range(coords.shape[1]):
            if float(df['likelihood'+str_i][j+1]) > .7:
                coords[i, j, :] = np.array((df['x'+str_i][j+1],
                df['y'+str_i][j+1]))
            else:
                coords[i,j,:] = np.nan
    return bp_list, coords

def get_3d_from_2d(arr, mtx, rvecs, tvecs):
    rotMatrix, jacobian = cv2.Rodrigues(rvecs)
    rotMatrix_inv = np.linalg.inv(rotMatrix)
    mtx_inv = np.linalg.inv(mtx)
    homogenous = np.ones((3,1))
    homogenous[0:2,:] = np.expand_dims(arr, 1)
    temp = np.dot(mtx_inv, homogenous)
    left = np.dot(rotMatrix_inv, temp)

    solution = np.subtract(left,np.dot(rotMatrix_inv, tvecs))
    
    return solution

def get_3d_from_2d_undistorted(arr, mtx, dist, rvecs, tvecs):
    rotMatrix, jacobian = cv2.Rodrigues(rvecs)
    rotMatrix_inv = np.linalg.inv(rotMatrix)
    
    arr_undistort = np.squeeze(cv2.undistortPoints(arr, mtx, dist))
    #mtx_inv = np.linalg.inv(mtx)
    homogenous = np.ones((3,1))
    homogenous[0:2,:] = np.expand_dims(arr_undistort, 1)
    #temp = np.dot(mtx_inv, homogenous)
    left = np.dot(rotMatrix_inv, homogenous)

    solution = np.subtract(left,np.dot(rotMatrix_inv, tvecs))
    
    return solution

def get_3d_dlc(arr, mtx, rvecs, tvecs):
    ret_arr = np.empty((arr.shape[0], arr.shape[1], 3))
    ret_arr[:,:,:] = np.nan
    


    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            temp = arr[i, j, :]
            if not np.isnan(temp).any():
                ret_arr[i, j, :] = np.squeeze(get_3d_from_2d(temp, mtx, rvecs, tvecs))

    return ret_arr

def get_3d_dlc_undistort(arr, mtx, dist, rvecs, tvecs):
    ret_arr = np.empty((arr.shape[0], arr.shape[1], 3))
    ret_arr[:,:,:] = np.nan
    


    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            temp = arr[i, j, :]
            if not np.isnan(temp).any():
                ret_arr[i, j, :] = np.squeeze(get_3d_from_2d_undistorted(temp, mtx, dist, rvecs, tvecs))

    return ret_arr

def angle(a, b, c):
    ba = a-b
    bc = c-b

    ret_arr = np.zeros(ba.shape[0])

    for idx in range(ba.shape[0]):
        cosine_angle = np.dot(ba[idx,:], bc[idx,:]) / (np.linalg.norm(ba[idx,:]) * np.linalg.norm(bc[idx,:]))
        angle = np.arccos(cosine_angle)
        ret_arr[idx] = np.degrees(angle)

    return ret_arr

def get_joint_angles(arr, list_of_joints):
    ret_arr = np.empty((arr.shape[1], len(list_of_joints)))
    ret_arr[:,:] = np.nan
    for idx, item in enumerate(list_of_joints):
        a = arr[item[0], :, :]
        b = arr[item[1],:,:]
        c = arr[item[2],:,:]
        
        ret_arr[:, idx] = angle(a,b,c)
    
    return ret_arr

def reshape_for_dlc(arr, numBodyParts, numFrames):
    large_list = []
    for i in range(numFrames):
        tiny_list = []
        for j in range(numBodyParts):
            tiny_list.append(arr[j, i, 0])
            tiny_list.append(arr[j, i, 1])
        large_list.append(tiny_list)
    
    return np.array(large_list)



def distance(p1, p2):
    
    squared_dist = np.sum((p1-p2)**2, axis=1)
    dist = np.sqrt(squared_dist)

    return dist


def compare_distances(arr1, arr2):
    ret_arr = np.empty((arr1.shape[0], arr1.shape[1]))
    ret_arr[:,:] = np.nan
    for idx, bodypart in enumerate(arr1):
        ret_arr[idx, :] = distance(arr1[idx,:,:], arr2[idx,:,:])
    
    return ret_arr

def trim_array(arr, curr_bp_list, wanted_bp_list):
    ret_arr = np.empty((len(wanted_bp_list), arr.shape[1], arr.shape[2]))
    ret_arr[:,:,:] = np.nan
    iterator = 0
    for idx, bp in enumerate(wanted_bp_list):
        found = curr_bp_list.index(bp)
        ret_arr[idx,:,:] = arr[found,:,:]
        
    return ret_arr

def skeleton_distances(arr, skeleton_list):
    ret_arr = np.empty((len(skeleton_list), arr.shape[1]))
    ret_arr[:,:] = np.nan
    for idx, skelly in enumerate(skeleton_list):
        bp_a = skelly[0]
        bp_b = skelly[1]
        
        ret_arr[idx, :] = distance(arr[bp_a, :,:], arr[bp_b,:,:])
    return ret_arr
        

def normalize_array(arr, maxs, mins):
    ret_arr = []
    for element in arr:
        temp = np.subtract(maxs, mins)
        test = np.divide(np.subtract(element, mins), temp)
        ret_arr.append(test)
    return np.array(ret_arr)
    
def extract_stupid(df):
    bp_list = list(dict.fromkeys(df.iloc[0]))
    df.drop(0, inplace=True)
    coords = np.zeros((len(bp_list), len(df.index), 2))
    for i in range(len(bp_list)):
        if i>0:
            str_i = '.' + str(i)
        else:
            str_i = ''
        for j in range(coords.shape[1]):
            coords[i, j, :] = np.array((df['x'+str_i][j+1],
            df['y'+str_i][j+1]))
    return bp_list, coords

def warp_and_concat(img_a, img_b, h_a, h_b):
    warp_a= cv2.warpPerspective(img_a, h_a, (512,512))
    warp_b= cv2.warpPerspective(img_b, h_b, (512,512))
    
    stitchy = np.concatenate((warp_a, warp_b), axis=1)
    
    return stitchy
