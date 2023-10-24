import traceback
import yaml
import gc

from src.neural_analysis import *
from src.wiener_filter import *
from src.filters import *
from src.folder_handler import *
from src.tdt_support import *
from src.decoders import *
from src.phase_decoder_support import *
from sklearn.metrics import r2_score
from scipy.signal import resample, find_peaks
from sklearn.decomposition import PCA
import scipy.io as sio
import cv2
import copy


class CortProcessor:
    '''
    class that can handle neural + kinematic/EMG data simultaneously
    upon initialization, extracts data from TDT and anipose file
    see cort_processor.md in 'docs' for more information
    '''
    
    def __init__(self, folder_path):
        if os.path.isdir(folder_path):
            # see docs/data_folder_layout.md for how to structure folder_path
            self.handler = FolderHandler(folder_path)
            self.tdt_data, self.kin_data = self.extract_data()
            
            self.crop_list = None
            crop = self.parse_config()
            if isinstance(crop, list):
                self.crop_list = crop
            
            self.data = {}
            self.data['rates'] = 'run_process_first'
            self.data['coords'] = 'run_process_first'
            self.data['angles'] = 'run_process_first'
            self.data['toe_height'] = 'run process first, then run process\
                    toehight'
            
            self.gait_indices = 'run get_gait_indices first'
            self.avg_gait_samples = 'run get_gait_indices first'
        
        else:
            print('this is filipe data i belive')
            self.handler = sio.loadmat(folder_path)
            self.gait_indices = None
            self.avg_gait_samples = None
            self.extract_filipe()
    
    def extract_filipe(self):
        self.data = {}
        # remoivng 0 rate channels, which
        # throws an error in wiener filter
        temp_rates = self.handler['SelectedSpikeData']
        mask = np.average(temp_rates, 0) > .1
        temp_rates_remove = temp_rates[:, mask]
        self.data['rates'] = [temp_rates_remove]
        
        self.data['coords'] = [self.handler['SelectedKinematicData']['kindata'][0, 0]]
        angles_temp = self.handler['SelectedKinematicData']['kinmeasures'][0, 0][0, 0]
        self.data['angles'] = [np.squeeze(np.array(angles_temp.tolist())).T]
        angle_names = ['ankle', 'limbfoot', 'hip', 'knee', 'toeheight']
        self.data['angle_names'] = angle_names
        
        temp_channels = self.handler['SelectedFieldDatastruct']['FieldCh2Use'][0][0][0].tolist()
        
        which_channels = copy.deepcopy(temp_channels)
        for idx in range(temp_rates.shape[1]):
            if np.average(temp_rates[:, idx]) < .1:
                baddy = temp_channels[idx]
                which_channels.remove(baddy)
        
        self.data['which_channels'] = which_channels
    
    #        samples = temp_rates.shape[0]
    #        num_channels = 32
    #        original_rates = np.zeros((samples, num_channels))
    #
    #        for idx in range(num_channels):
    #            if idx in which_channels:
    #                position = np.where(which_channels==idx)[0][0]
    #                if np.average(temp_rates[:, position]) > .1:
    #                    original_rates[:, idx] = temp_rates[:, position]
    #                else:
    #                    original_rates[:,idx] = np.nan
    #            else:
    #                original_rates[:,idx] = np.nan
    #
    #        self.data['original_rates'] = [original_rates]
    #
    def parse_config(self):
        """
        this loads config.yaml file
        """
        try:
            path = self.handler.folder_path
            with open(f"{path}/config.yaml", "r") as stream:
                yaml_stuff = yaml.safe_load(stream)
                crop = 'crop'
                
                if crop in yaml_stuff:
                    return yaml_stuff[crop]
                else:
                    print('no crop')
                    return 0
        except yaml.YAMLError as exc:
            print(exc)
    
    def extract_data(self):
        """
        this is called when a CortProcessor is initialized.
        It extracts raw neural, raw angles, and raw coords
        neural data is saved under CortProcessr.tdt_data
        kinematic data is saved under CortProcessor.kin_data

        If you use process cort script, all these attriibutes are overwritten
        since raw data isnt saved after processing.
        """
        print('Extracting data')
        tdt_data_list = []
        raw_ts_list = self.handler.ts_list
        raw_tdt_list = self.handler.tdt_list
        for i in range(len(raw_tdt_list)):
            tdt_data_list.append(extract_tdt(raw_tdt_list[i], raw_ts_list[i]))
        
        kin_data_list = []
        
        raw_coords_list = self.handler.coords_list
        raw_angles_list = self.handler.angles_list
        
        for i in range(len(raw_coords_list)):
            kin_data_list.append(extract_kin_data(raw_coords_list[i], raw_angles_list[i]))
        
        return tdt_data_list, kin_data_list
    
    def process(self, manual_crop_list=None, threshold_multiplier=-3.0, binsize=0.05):
        """
        1) takes raw neural data, then bandpass+notch filters, then extracts
        spikes using threshold_multiplier, then bins into firing rates with bin
        size
        2) takes coordinates/angles, then resamples at 20 HZ
        3) saves all under CortHandler.data

        crop is automatically read from config.yaml file, unless manual crop
        list is set.
        """
        
        print('Processing data')
        if self.crop_list is not None:
            crop_list = self.crop_list
        elif manual_crop_list is not None:
            crop_list = manual_crop_list
        else:
            # TODO
            print('need to make it so that it is uncropped')
        
        tdt_data_list = self.tdt_data
        kin_data_list = self.kin_data
        
        self.data = {}
        
        self.data['bodyparts'] = kin_data_list[0]['bodyparts']
        self.data['angle_names'] = kin_data_list[0]['angles_list']
        
        self.data['rates'] = []
        self.data['coords'] = []
        self.data['angles'] = []
        for i in range(len(tdt_data_list)):
            # syncs neural data/kinematic data, then crops
            self.tdt_data[i], self.kin_data[i] = self.crop_data(tdt_data_list[i], kin_data_list[i], crop_list[i])
            
            fs = self.tdt_data[i]['fs']  # quick accessible variable
            neural = self.tdt_data[i]['neural']  # quick accessible variable
            
            # notch and bandpass filter
            
            ##common mode rejection assumes equal impedence and is therefore not recommended 
            # means = np.mean(neural, axis = 1)
            # commonmode = []
            # for j in range(neural.shape[1]):
            #     commonhold = neural[:,j] - means
            #     commonmode.append(commonhold)
            # commonmode = np.array(commonmode).T
            
            half_filtered_neural = fresh_filt(neural, 350, 8000, fs, order=4)
            filtered_neural = notch_filter(half_filtered_neural, fs)
            clean_filtered_neural = remove_artifacts(filtered_neural, fs)
            
            # extract spike, impose refeactory limit (post artifact rejection) and bin
            spikes_tmp = threshold_crossings_refrac(clean_filtered_neural, threshold_multiplier)
            spikes = refractory_limit(spikes_tmp, fs)
            firing_rates = spike_binner(spikes, fs, binsize)
            
            self.data['rates'].append(firing_rates)
            
            temp_angles = self.kin_data[i]['angles']  # quick accessible variable
            temp_coords = self.kin_data[i]['coords']
            
            # resample at same frequency of binned spikes
            resampled_angles = resample(temp_angles, firing_rates.shape[0], axis=0)
            resampled_coords = resample(temp_coords, firing_rates.shape[0], axis=0)
            
            self.data['angles'].append(resampled_angles)
            self.data['coords'].append(resampled_coords)
            
            # remove raw data to save memory
            self.tdt_data[i] = 0
            self.kin_data[i] = 0
            gc.collect()
        
        # returning stitched rates --- we don't directly use this for anything.
        return np.vstack(self.data['rates']), np.vstack(self.data['angles'])

    def save_processed(self, save_path, save_name='processed_data'):
        '''
        saves processed data into a pickle file
        '''
        save_path = f'{save_path}/{save_name}.pickle'
        with open(save_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print(f'Processed data saved to {save_path}')
        return 0
            
    def process_toe_height(self):
        """
        extracts toe height from data['coords'], then scales such that lowest
        toe height is 0.

        toe_num is which bodypart in data['coords'] is the toe. default is
        bodypart 0
        """
        try:
            toe_num = self.bodypart_helper('toe')
            temp_list = []
            self.data['toe_height'] = []
            minimum_toe = 1000  # arbitrary high number to be beat
            for i in range(len(self.data['coords'])):
                # find y-value of toe
                toe_height = self.data['coords'][i][:, toe_num, 1]
                if np.min(toe_height) < minimum_toe:
                    minimum_toe = np.min(toe_height)
                temp_list.append(toe_height)
            for i in range(len(temp_list)):
                norm_toe_height = temp_list[i] - minimum_toe
                self.data['toe_height'].append(norm_toe_height)
            
            return np.hstack(self.data['toe_height'])
        except Exception as e:
            print('failed!! did you run process first')
            print(e)
    
    def crop_data(self, tdt_data, kin_data, crop):
        '''
        helper function that both syncs and crops neural/kinematic data
        see docs/cam_something.odg for how it syncs em.
        '''
        
        crop_tdt_datafile = tdt_data
        crop_kin_datafile = kin_data
        
        start_time = crop[0]
        end_time = crop[1]
        
        kin_start = start_time * 200
        kin_end = end_time * 200
        
        init_start_sample = get_sync_sample(tdt_data)
        
        start_sample = round(init_start_sample + (start_time * tdt_data['fs']))
        end_sample = round(init_start_sample + (end_time * tdt_data['fs']))
        
        temp_neural = tdt_data['neural']  # going to slice variable
        temp_ts = tdt_data['ts']
        
        crop_tdt_datafile['neural'] = temp_neural[start_sample:end_sample, :]
        crop_tdt_datafile['ts'] = temp_ts[start_sample:end_sample]
        
        del (temp_neural)
        del (temp_ts)
        
        gc.collect()
        
        temp_coords = kin_data['coords']
        temp_angles = kin_data['angles']
        
        crop_kin_datafile['coords'] = temp_coords[kin_start:kin_end, :, :]
        crop_kin_datafile['angles'] = temp_angles[kin_start:kin_end, :]
        # maybe need edge-case if only single angle/bodypart
        
        return crop_tdt_datafile, crop_kin_datafile
    
    def stitch_and_format(self, firing_rates_list=None, resampled_angles_list=None):
        """
        takes list of rates, list of angles, then converts them into lags of 10
        using format rate in wiener_filter.py, and then stitches them into one
        big array

        both lists must have same # of elements, and each array inside list
        must have the same size as the corresponding array in the other list.
        """
        if firing_rates_list == None and resampled_angles_list == None:
            firing_rates_list = self.data['rates']
            resampled_angles_list = self.data['angles']
        
        assert isinstance(firing_rates_list, list), 'rates must be list'
        assert isinstance(resampled_angles_list, list), 'angles must be list'
        formatted_rates = []
        formatted_angles = []
        
        for i in range(len(firing_rates_list)):
            f_rate, f_angle = format_data(firing_rates_list[i], resampled_angles_list[i])
            formatted_rates.append(f_rate)
            formatted_angles.append(f_angle)
        
        if len(formatted_rates) == 1:  # check if just single array in list
            rates = np.array(formatted_rates)
        else:  # if multiple, stitch into single array
            rates = np.vstack(formatted_rates)
        
        if len(formatted_angles) == 1:  # check if single array
            kin = np.array(formatted_angles)
        elif formatted_angles[0].ndim > 1:  # check if multiple angles
            kin = np.vstack(formatted_angles)
        else:
            kin = np.hstack(formatted_angles)
        
        return np.squeeze(rates), np.squeeze(kin)
    
    def stitch_data(self, firing_rates_list, resampled_angles_list):
        """
        deprecated. you can just use np.vstack instead of this
        WARNING: VSTACK WILL NOT BE INDEXED AT PARITY UNLESS THE LAST 10 ENTRIES OF EACH LIST IS OMITTED
        """
        rates = np.vstack(firing_rates_list)
        kin = np.vstack(resampled_angles_list)
        
        return rates, kin
    
    def spectro1(self, window=4, rates=None, plotting=False):
        """
        this makes spectrograms!
        currently there is no normalization between channels
        rate stacking is sliced for index parity with stitch and format
        """
        if rates == None:
            rate_append = []
            for i in range(len(self.data['rates'])):
                rate_append.append(self.data['rates'][i][10:])
            rates = np.vstack(rate_append)
        seconds = window
        fsr = 20
        tlim = (rates.shape[0] * 50) / 1000
        nperseg_rates = int(seconds * fsr)
        Sxx_list = []
        for i in range(0, 32):
            f, t, Sxx = signal.spectrogram(rates[:, i], fs=fsr, nperseg=nperseg_rates)
            Sxx_list.append(Sxx)
        Sxx_sum = np.sum(Sxx_list, axis=0)
        if plotting == True:
            fig, (ax2) = plt.subplots(1, 1, figsize=(10, 6))
            ax2.pcolormesh(t, f, Sxx_sum, cmap='terrain', shading='gouraud')
            ax2.set_ylim([0, 10])
            # ax2.set_xlim([0,tlim])
            ax2.set_ylabel('Frequency [Hz]')
            ax2.set_xlabel('Time [sec]')
            ax2.set_title('spike rate spectrogram')
        return Sxx_sum, t, f
    
    def subsample(self, percent, X=None, Y=None):
        """
        function to subsample RAW data (not gait aligned).
        we keep it in order since we go by folds.
        """
        if X is None:
            X = self.data['rates']
        if Y is None:
            Y = self.data['angles']
        
        if percent == 1.0:
            return X, Y
        
        new_x = []
        new_y = []
        
        for i in range(len(X)):
            subsample = int(np.round(percent * X[i].shape[0]))
            new_x.append(X[i][:subsample, :])
            new_y.append(Y[i][:subsample, :])
        
        return new_x, new_y
    
    def decode_angles(self, X=None, Y=None, metric_angle='limbfoot', scale=False):
        """
        takes list of rates, angles, then using a wiener filter to decode. 
        if no parameters are passed, uses data['rates'] and data['angles']

        returns best_h, vaf (array of all angles and all folds), test_x (the
        x test set that had best performance, and test_y (the y test set that
        had best formance)
        """
        try:
            if X is None:
                X = self.data['rates']
            if Y is None:
                Y = self.data['angles']
            
            if scale is True:
                X = self.apply_scaler(X)
            X, Y = self.stitch_and_format(X, Y)
            
            metric = self.angle_name_helper(metric_angle)
            
            h_angle, vaf_array, final_test_x, final_test_y = decode_kfolds(X, Y, metric=metric)
            
            self.h_angle = h_angle
            return h_angle, vaf_array, final_test_x, final_test_y
        except Exception as e:
            print(e)
            print('did you run process() first.')
    
    def decode_toe_height(self):
        '''
        a PIPELINE code, which you can you only after you run
        process_toe_height. You cannot pass any custom parameters in this. it
        uses a wiener filter to create a decoder just for toe height
        '''
        try:
            X, Y = self.stitch_and_format(self.data['rates'], self.data['toe_height'])
            h_toe, vaf_array, final_test_x, final_test_y = decode_kfolds_single(X, Y)
            self.h_toe = h_toe
            return h_toe, vaf_array, final_test_x, final_test_y
        except:
            print('did you run process_toe_height() yet?????')
    
    def decode_phase(self, rates=None, angles=None, metric_angle='limbfoot'):
        if rates is None and angles is None:
            full_rates, full_angles = self.stitch_and_format(self.data['rates'], self.data['angles'])
        
        elif isinstance(rates, list):
            full_rates, full_angles = self.stitch_and_format(rates, angles)
        else:
            full_rates = rates
            full_angles = angles
        
        angle_number = self.angle_name_helper(metric_angle)
        phase_list = []
        
        for i in range(full_angles.shape[1]):
            peak_list = tailored_peaks(full_angles, i, self.data['angle_names'][i])
            phase_list_tmp = to_phasex(peak_list, full_angles[:, i])
            phase_list.append(phase_list_tmp)
        phase_list = np.array(phase_list).T
        sin_array, cos_array = sine_and_cosine(phase_list)
        h_sin, vaf_sin, test_sinx, test_siny = decode_kfolds(X=full_rates, Y=sin_array, metric=angle_number,
                                                             vaf_scoring=False)
        h_cos, vaf_cos, test_cosx, test_cosy = decode_kfolds(X=full_rates, Y=cos_array, metric=angle_number,
                                                             vaf_scoring=False)
        predicted_sin = predicted_lines(full_rates, h_sin)
        predicted_cos = predicted_lines(full_rates, h_cos)
        
        # test_sin = predicted_lines(test_sinx, h_sin)
        # test_cos = predicted_lines(test_cosx, h_cos)
        # test_predic_arctans = arctan_fn(test_sin, test_cos)
        # test_real_arctans = arctan_fn(test_siny, test_cosy)
        
        arctans = arctan_fn(predicted_sin, predicted_cos)
        # r2_array = []
        # for i in range(sin_array.shape[1]):
        #    r2_sin = r2_score(sin_array[:,i], predicted_sin[:,i])
        #    r2_cos = r2_score(cos_array[:,i], predicted_cos[:,i])
        #    r2_array.append(np.mean((r2_sin,r2_cos)))
        self.phase_list = phase_list
        self.h_sin = h_sin
        self.h_cos = h_cos
        
        return arctans, phase_list, (vaf_sin, vaf_cos), h_sin, h_cos, sin_array, cos_array
    
    def get_H(self, H):
        if H == 'toe':
            H_mat = self.h_toe
        if H == 'angle':
            H_mat = self.h_angle
        if H == 'cos':
            H_mat = self.h_cos
        if H == 'sin':
            H_mat = self.h_sin
        return H_mat
    
    def impulse_response(self, AOI, H=None, plotting=True):
        phase_list = self.phase_list
        if H == None:
            H = 'sin'
        h_mat = self.get_H(H)
        response = impulse_response(AOI, h_mat, phase_list, plotting)
        return response
    
    def get_gait_indices(self, Y=None, metric_angle='limbfoot'):
        if Y is None:
            Y_ = self.data['angles']
        else:
            Y_ = Y
            assert isinstance(Y, list), 'Y must be a list'
        
        gait_indices = []
        samples_list = []
        
        angle_number = self.angle_name_helper(metric_angle)
        
        for angle in Y_:
            peaks = tailored_peaks(angle, angle_number, metric_angle)
            
            gait_indices.append(peaks)
            samples_list.append(np.diff(peaks))
        
        if len(samples_list) > 1:
            samples = np.concatenate(samples_list)
        else:
            samples = samples_list[0]
        
        avg_gait_samples = int(np.round(np.average(samples)))
        
        if Y is None:
            self.gait_indices = gait_indices
            self.avg_gait_samples = avg_gait_samples
        
        return gait_indices, avg_gait_samples
    
    def deprec_get_gait_indices(self, Y=None, metric_angle='limbfoot'):
        '''
        This takes a kinematic variable, and returns indices where each peak is
        found. It also returns the average number of samples between each
        peaks. 

        If passing in without parameter, it uses the 3rd angle measurement,
        which is usually the limbfoot angle. 

        This is mainly used as a starter method for other method.
        Divide_into_gaits for instance takes in these indices, and divides both
        the kinematics and rates into gait cycles.
        '''
        if Y is None:
            Y_ = self.data['angles']
        else:
            assert isinstance(Y, list), 'Y must be a list'
            Y_ = Y
        
        angle_number = self.angle_name_helper(metric_angle)
        
        gait_indices = []
        samples_list = []
        for angle in Y_:
            if angle.ndim > 1:
                angle = angle[:, angle_number]
            temp_peaks, nada = find_peaks(angle, prominence=10, distance=5)
            avg_ = np.average(angle[temp_peaks])
            std_ = np.std(angle[temp_peaks])
            
            temp2_peaks = temp_peaks[np.argwhere(angle[temp_peaks] < avg_ + 30)]
            temp2_peaks = np.squeeze(temp2_peaks)
            peaks = temp2_peaks[np.argwhere(angle[temp2_peaks] > avg_ - 30)]
            peaks = np.squeeze(peaks)
            peaks = np.append(peaks, np.size(angle) - 1)
            peaks = np.insert(peaks, 0, 0)
            gait_indices.append(peaks)
            samples_list.append(np.diff(peaks))
        
        if len(samples_list) > 1:
            samples = np.concatenate(samples_list)
        else:
            samples = samples_list[0]
        
        avg_gait_samples = int(np.round(np.average(samples)))
        
        if Y is None:
            self.gait_indices = gait_indices
            self.avg_gait_samples = avg_gait_samples
        
        return gait_indices, avg_gait_samples
    
    def divide_into_gaits(self, X=None, Y=None, gait_indices=None, avg_gait_samples=None, bool_resample=True):
        '''
        this takes in X, which is usually rates, Y which is usually some
        kinematic variable, and indices, which tell you how to divide up the
        data, and divides all the data up as lists. Since the originall X is
        already a list, it returns a list of list of lists. Confusing?
        
        if you don't pass any parameters, then get_gait_indices must be run
        first. This finds gait_indices/avg_gait_samples using limbfoot angle.

        If you don't pass in X/Y paramters, it by default uses rates and
        angles, and then divides em up.
        '''
        
        if gait_indices is None:
            gait_indices = self.gait_indices
        
        if avg_gait_samples is None:
            avg_gait_samples = self.avg_gait_samples
        
        if X is None:
            rates = self.data['rates']
        else:
            assert isinstance(X, list), 'X must be a list'
            rates = X
        
        if Y is None:
            angles = self.data['angles']
        else:
            assert isinstance(Y, list), 'Y must be a list'
            angles = Y
        
        X_gait = []
        Y_gait = []
        
        for i, trial_gait_index in enumerate(gait_indices):
            trial_rate_gait = []
            trial_angle_gait = []
            for j in range(np.size(trial_gait_index) - 1):
                end = trial_gait_index[j + 1]
                start = trial_gait_index[j]
                
                temp_rate = rates[i][start:end, :]
                temp_angle = angles[i][start:end, :]
                
                if bool_resample:
                    temp_rate = resample(temp_rate, avg_gait_samples, axis=0)
                    temp_angle = resample(temp_angle, avg_gait_samples, axis=0)
                
                trial_rate_gait.append(temp_rate)
                trial_angle_gait.append(temp_angle)
            X_gait.append(trial_rate_gait)
            Y_gait.append(trial_angle_gait)
        
        if bool_resample:
            X_gait = np.array(X_gait)
            Y_gait = np.array(Y_gait)
        
        if X is None:
            self.rates_gait = X_gait
        if Y is None:
            self.angles_gait = Y_gait
        
        return X_gait, Y_gait  # return list of list of lists lol
    
    def toe_to_stance_swing(self, toe_height):
        peaks, _ = find_peaks(toe_height, height=12)
        peaks = np.append(peaks, np.size(toe_height))
        peaks = np.insert(peaks, 0, 0)
        ss_list = []
        
        for i in range(np.size(peaks) - 1):
            end = peaks[i + 1]
            start = peaks[i]
            
            gait = toe_height[start:end]
            dx = np.gradient(gait)
            ddx = np.gradient(dx)
            
            ddx_peaks, _ = find_peaks(ddx, height=0.02)
            if np.size(ddx_peaks) == 2:
                ss = np.ones(np.size(gait), dtype=bool)
                ss[ddx_peaks[0]:ddx_peaks[1]] = 0
            else:
                minny = np.amin(gait)
                ss = gait > minny + 2
            
            ss_list.append(ss)
        
        stance_swing = np.hstack(ss_list)
        return stance_swing
    
    def get_stance_swing(self):
        if self.data['toe_height'] is None:
            output = 'run process toe height first'
        
        toe_list = self.data['toe_height']
        stance_swing_list = []
        
        for toe in toe_list:
            stance_swing_list.append(self.toe_to_stance_swing(toe))
        
        rates = self.data['rates']
        
        X, Y = self.stitch_and_format(rates, stance_swing_list)
        return X, Y
    
    def remove_bad_gaits(self, X=None, Y=None, gait_indices=None, avg_gait_samples=None, bool_resample=True):
        '''
        similar to divide into gaits, but instead of just dividing, it also
        removes any gait cycles that have a much smaller or much larger amount
        of samples. in a sense it divies up gaits and removes bad ones.

        must run get gait indices first!
        '''
        if gait_indices is None:
            gait_indices = self.gait_indices  # print(self.gait_indices)
        
        if avg_gait_samples is None:
            avg_gait_samples = self.avg_gait_samples
        
        if X is None:
            rates = self.data['rates']
        elif isinstance(X, list):
            rates = X
        else:
            print('X must be list')
            return
        
        if Y is None:
            angles = self.data['angles']
        elif isinstance(Y, list):
            angles = Y
        else:
            print('Y must be list')
            return
        
        # above = slow gait cycles, below = too fast gait cycles
        above = 1.33 * avg_gait_samples
        below = .66 * avg_gait_samples
        bads_list = []
        for idx in gait_indices:
            # we are iterating and adding to a list any too slow or fast gaits
            bad_above = np.argwhere(np.diff(idx) > above)
            bad_below = np.argwhere(np.diff(idx) < below)
            
            bads_list.append(np.squeeze(np.concatenate(
                (bad_above, bad_below))).tolist())  # bad_list now has all the indices of bad gait cycles!
        proc_rates = []
        proc_angles = []
        for i, trial_indices in enumerate(gait_indices):
            # trial_rate_gait = []
            # trial_angle_gait = []
            for j in range(np.size(trial_indices) - 1):
                if j in bads_list[i]:
                    # skip if its a bad gait cycle!
                    continue
                else:
                    end = trial_indices[j + 1]
                    start = trial_indices[j]
                    
                    temp_rate = rates[i][start:end, :]
                    temp_angle = angles[i][start:end, :]
                    
                    low_number_check = temp_angle[:, 6] > 100
                    if False in low_number_check:
                        continue
                    if bool_resample:
                        temp_rate = resample(temp_rate, avg_gait_samples, axis=0)
                        temp_angle = resample(temp_angle, avg_gait_samples, axis=0)
                    proc_rates.append(temp_rate)
                    proc_angles.append(temp_angle)
            
            # proc_rates.append(trial_rate_gait)  # proc_angles.append(trial_angle_gait)
        
        return np.vstack(proc_rates), np.vstack(proc_angles)
    
    def neuron_tuning(self, rates_gait=None):
        '''
        this takes in neural data that is divided into gaits, and sorts each
        channel by where in the gait cycle each channel is most active

        feed the output of this into plot_raster!
        '''
        try:
            if rates_gait is None:
                rates_gait = self.rates_gait
            rates_gait = np.vstack(rates_gait)
            gait_array_avg = np.average(rates_gait, axis=0)
            df = pd.DataFrame(gait_array_avg)
            temp = df.iloc[:, df.idxmax(axis=0).argsort()]
            self.norm_sorted_neurons = ((temp - temp.min()) / (temp.max() - temp.min()))
            return self.norm_sorted_neurons
        
        except Exception as e:
            print(e)
            print('make sure you run divide into gaits first')
    
    def convert_to_phase(self, gait_indices=None):
        '''
        this is still TODO. might already work, but double check this
        '''
        # TODO
        phase_list = []
        
        if gait_indices is None:
            gait_indices = self.gait_indices
        for i in range(np.size(gait_indices) - 1):
            end = gait_indices[i + 1]
            start = gait_indices[i]
            phase = np.sin(np.linspace(0.0, 2.0 * math.pi, num=end - start, endpoint=False))
            phase_list.append(phase)
        
        return phase_list  # use np.hstack on output to get continuous
    
    def apply_PCA(self, dims=None, X=None, transformer=None):
        '''
        this works now?
        '''
        if X is None:
            X = self.data['rates']
        if dims is None:
            dims = .95
        X_full = np.vstack(X)
        if transformer is None:
            pca_object = PCA(n_components=dims)
            pca_object.fit(X_full)
        else:
            pca_object = transformer
        
        x_pca = []
        
        for X_recording in X:
            x_pca.append(pca_object.transform(X_recording))
        
        self.num_components = x_pca[0].shape[1]
        self.pca_object = pca_object
        return x_pca
    
    def apply_scaler(self, X=None, scaler=None):
        if X is None:
            X = self.data['rates']
        if scaler is None:
            my_scaler = StandardScaler()
            X_full = np.vstack(X)
            my_scaler.fit(X_full)
        else:
            my_scaler = scaler
        
        scaled_X = []
        for X_recording in X:
            scaled_X.append(my_scaler.transform(X_recording))
        
        self.scaler = my_scaler
        return scaled_X
    
    def display_frame(self, video_path, recording_number, sample_number):
        cap = cv2.VideoCapture(video_path)
        crop_times = self.crop_list[recording_number]
        first_frame = crop_times[0] * 200
        my_frame = first_frame + (sample_number * 4)
        cap.set(1, my_frame)
        ret, frame = cap.read()
        
        video_time = my_frame / 200
        print(f'video_time is: {video_time}')
        
        return frame
    
    def predicted_lines_malleable(self, h_sin, h_cos):
        try:
            full_rates, full_angles = self.stitch_and_format(self.data['rates'], self.data['angles'])
            phase_list = self.phase_list
            sin_array, cos_array = sine_and_cosine(phase_list)
            predicted_sin = predicted_lines(full_rates, h_sin)
            predicted_cos = predicted_lines(full_rates, h_cos)
            arctans = arctan_fn(predicted_sin, predicted_cos)
            r2_array = []
            for i in range(sin_array.shape[1]):
                r2_sin = r2_score(sin_array[:, i], predicted_sin[:, i])
                r2_cos = r2_score(cos_array[:, i], predicted_cos[:, i])
                r2_array.append(np.mean((r2_sin, r2_cos)))
            return arctans, phase_list, r2_array
        except:
            print('error lol')
            print('Feed in sin and cos H matricies from another session to test gernealizability')
    
    def angle_name_helper(self, angle_name):
        return self.data['angle_names'].index(angle_name)
    
    def bodypart_helper(self, bodypart):
        return self.data['bodyparts'].index(bodypart)
