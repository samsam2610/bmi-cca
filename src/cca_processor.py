from src.neural_analysis import *
from src.cort_processor import *
from src.wiener_filter import *
from src.folder_handler import *
from src.tdt_support import *
from src.decoders import *

from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold


class CCAProcessor:
    def __init__(self, cp1, cp2, metric_angle='limbfoot', align=0):
        # align = 0 is sorting and stitching
        # align = 1 is resampling
        self.cp1 = cp1
        # angle_number1 = cp1.angle_name_helper(metric_angle)
        self.cp1.get_gait_indices(metric_angle=metric_angle)
        
        self.cp2 = cp2
        ##angle_number2 = cp2.angle_name_helper(metric_angle)
        self.cp2.get_gait_indices(metric_angle=metric_angle)
        
        self.metric_angle = metric_angle
        
        self.data = {'cp1': {}, 'cp2': {}}
        
        self.data['cp1']['pca_x'] = None
        self.data['cp2']['pca_x'] = None
        
        if align == 0:
            
            self.data['cp1']['proc_x'], self.data['cp1']['proc_y'], self.data['cp2']['proc_x'], self.data['cp2'][
                'proc_y'] = self.sort_and_align()
        
        elif align == 1:
            
            self.data['cp1']['proc_x'], self.data['cp1']['proc_y'], self.data['cp2']['proc_x'], self.data['cp2'][
                'proc_y'] = self.process_and_align_kinematics()
        
        # self.data['cp1']['h'], self.data['cp1']['proc_vaf'], nada, nada =\
        # self.cp1.decode_angles(X=[self.data['cp1']['proc_x']],\
        # Y=[self.data['cp1']['proc_y']])
        
        # self.data['cp2']['h'], self.data['cp2']['proc_vaf'], nada, nada =\
        # self.cp2.decode_angles(X=[self.data['cp2']['proc_x']],\
        # Y=[self.data['cp2']['proc_y']])
        
        print(self.data['cp1']['proc_x'].shape)
        print(self.data['cp1']['proc_y'].shape)
        print(self.data['cp2']['proc_x'].shape)
        print(self.data['cp2']['proc_y'].shape)
    
    def get_better_decoder(self, metric_angle=None):
        if metric_angle is None:
            metric_angle = self.cp1.metric_angle
        
        angle_number = self.cp1.angle_name_helper(metric_angle)
        
        nada, vaf1, nada, nada = self.cp1.decode_angles()
        nada, vaf2, nada, nada = self.cp2.decode_angles()
        
        if np.average(vaf1, 1)[angle_number] >= np.average(vaf2, 1)[angle_number]:
            print('cp1 is better')
        else:
            print('cp2 is better')
    
    def check_same_kinematics(self):
        if self.cp1.data['angle_names'] == self.cp2.data['angle_names']:
            output = 'should be good to align'
            return output
        else:
            output = 'kinematics are different'
            return output
    
    def apply_PCA(self, cp1_x=None, cp2_x=None, preset_num_components=None, day_0_transformer=None):
        
        if cp1_x is None:
            cp1_x = self.data['cp1']['proc_x']
        if cp2_x is None:
            cp2_x = self.data['cp2']['proc_x']
        
        if preset_num_components is None:
            pca_cp1 = PCA(n_components=.95)
            cp1_pca = pca_cp1.fit_transform(self.data['cp1']['proc_x'])
            
            pca_cp2 = PCA(n_components=.95)
            cp2_pca = pca_cp2.fit_transform(self.data['cp2']['proc_x'])
            
            num_components = min(cp2_pca.shape[1], cp1_pca.shape[1])
            
            self.num_components = num_components
        else:
            num_components = min(len(cp2_x), len(cp1_x), preset_num_components)
            self.num_components = num_components
        
        # Fix issue with scipy and numpy if cp2 x data is too few will result in error
        # try pca solver for cp2 first, if succeeded then continue, if not, remove minimal n-components requirement
        
        try:
            pca_cp2 = PCA(n_components=num_components)
            self.data['cp2']['pca_x'] = pca_cp2.fit_transform(cp2_x)
            
            if day_0_transformer is None:
                pca_cp1 = PCA(n_components=num_components)
            else:
                pca_cp1 = day_0_transformer
        except:
            pca_cp2 = PCA()
            print('Auto PCA failed, removing n-component requirement')
            self.data['cp2']['pca_x'] = pca_cp2.fit_transform(cp2_x)
            
            if day_0_transformer is None:
                pca_cp1 = PCA()
            else:
                pca_cp1 = day_0_transformer
        
        if day_0_transformer is None:
            self.data['cp1']['pca_x'] = pca_cp1.fit_transform(cp1_x)
        else:
            self.data['cp1']['pca_x'] = pca_cp1.transform(cp1_x)
        
        self.data['cp1']['pca_transformer'] = pca_cp1
        self.data['cp2']['pca_transformer'] = pca_cp2
        
        return self.data['cp1']['pca_x'], self.data['cp2']['pca_x']
    
    def apply_CCA_clean(self, cp1_x=None, cp2_x=None, transformer=None):
        ''' deprecated i think'''
        
        if cp1_x is None:
            if transformer is None:
                cp1_x = self.data['cp1']['proc_x']
        if cp2_x is None:
            cp2_x = self.data['cp2']['proc_x']
        
        num_components = cp2_x.shape[0]
        
        if transformer is None:
            cca_cp1cp2 = CCA(n_components=num_components, scale=False)
            x1_cca, x2_cca = cca_cp1cp2.fit_transform(cp1_x, cp2_x)
        else:
            cca_cp1cp2 = transformer
            nada, x2_cca = cca_cp1cp2.transform(cp2_x, cp2_x)
        
        self.cca = cca_cp1cp2
        self.x2_cca = x2_cca
        
        x2_into_x1 = cca_cp1cp2.inverse_transform(x2_cca)
        
        return cca_cp1cp2, x2_into_x1
    
    def apply_CCA(self, cp1_x=None, cp2_x=None, preset_num_components=None, transformer=None, pca=False):
        if cp1_x is None:
            if transformer is None:
                if pca:
                    cp1_x = self.data['cp1']['pca_x']
                else:
                    cp1_x = self.data['cp1']['proc_x']
        if cp2_x is None:
            if pca:
                cp2_x = self.data['cp2']['pca_x']
            else:
                cp2_x = self.data['cp2']['proc_x']
        if pca:
            if preset_num_components is None:
                num_components = 8
            else:
                num_components = min(len(cp2_x), len(cp1_x), preset_num_components)
        else:
            num_components = cp2_x.shape[1]
        
        if transformer is None:
            
            cca_cp1cp2 = CCA(n_components=num_components, scale=False)
            x1_cca, x2_cca = cca_cp1cp2.fit_transform(cp1_x, cp2_x)
        
        else:
            cca_cp1cp2 = transformer
            x1_cca, x2_cca = cca_cp1cp2.transform(cp2_x, cp2_x)
        
        self.cca = cca_cp1cp2
        self.x2_cca = x2_cca
        
        # for i in range(3):
        #     corr = np.around(np.corrcoef(x1_cca[:, i], x2_cca[:, i])[0,1], 2)
        #     print(f'dim{i} corr is {corr}')
        x2_into_x1 = cca_cp1cp2.inverse_transform(x2_cca)
        
        return cca_cp1cp2, x2_into_x1
    
    def process_and_align_kinematics(self):
        
        print(self.check_same_kinematics())
        
        avg_samples = self.cp2.avg_gait_samples  # arbitrarily grab from 1
        cp1_gait_x, cp1_gait_y = self.cp1.remove_bad_gaits(avg_gait_samples=avg_samples)
        cp2_gait_x, cp2_gait_y = self.cp2.remove_bad_gaits()
        
        if len(cp1_gait_x) >= len(cp2_gait_x):
            end_slice = len(cp2_gait_x)
            cp1_gait_x = cp1_gait_x[:end_slice]
            cp1_gait_y = cp1_gait_y[:end_slice]
        
        else:
            end_slice = len(cp1_gait_x)
            cp2_gait_x = cp2_gait_x[:end_slice]
            cp2_gait_y = cp2_gait_y[:end_slice]
        
        total_samples = cp1_gait_x.shape[0] * cp1_gait_x.shape[1]
        
        cp1_gait_x = np.reshape(cp1_gait_x, (total_samples, cp1_gait_x.shape[2]))
        cp1_gait_y = np.reshape(cp1_gait_y, (total_samples, cp1_gait_y.shape[2]))
        cp2_gait_x = np.reshape(cp2_gait_x, (total_samples, cp2_gait_x.shape[2]))
        cp2_gait_y = np.reshape(cp2_gait_y, (total_samples, cp2_gait_y.shape[2]))
        
        return cp1_gait_x, cp1_gait_y, cp2_gait_x, cp2_gait_y
    
    def sort_and_align(self, sample_variance=5):
        
        print(self.check_same_kinematics())
        
        cp1_gait_x, cp1_gait_y = self.cp1.divide_into_gaits(bool_resample=False)
        
        cp2_gait_x, cp2_gait_y = self.cp2.divide_into_gaits(bool_resample=False)
        
        cp1_gait_x = cp1_gait_x[0]
        cp1_gait_y = cp1_gait_y[0]
        cp2_gait_x = cp2_gait_x[0]
        cp2_gait_y = cp2_gait_y[0]
        
        cp2_avg_gaits = self.cp2.avg_gait_samples
        
        cp1_x_sortdict = {}
        cp1_y_sortdict = {}
        cp2_x_sortdict = {}
        cp2_y_sortdict = {}
        
        var_range = range(cp2_avg_gaits - sample_variance, cp2_avg_gaits + sample_variance + 1)
        
        for i in var_range:
            cp1_x_sortdict[i] = []
            cp1_y_sortdict[i] = []
            cp2_x_sortdict[i] = []
            cp2_y_sortdict[i] = []
            for idx in range(len(cp1_gait_x)):
                if len(cp1_gait_x[idx]) == i:
                    cp1_x_sortdict[i].append(cp1_gait_x[idx])
                    cp1_y_sortdict[i].append(cp1_gait_y[idx])
            for idx in range(len(cp2_gait_x)):
                if len(cp2_gait_x[idx]) == i:
                    cp2_x_sortdict[i].append(cp2_gait_x[idx])
                    cp2_y_sortdict[i].append(cp2_gait_y[idx])
        
        cp1_final_x = []
        cp1_final_y = []
        cp2_final_x = []
        cp2_final_y = []
        for i in var_range:
            num = min(len(cp1_x_sortdict[i]), len(cp2_x_sortdict[i]))
            
            for j in range(num):
                cp1_final_x.append(cp1_x_sortdict[i][j])
                cp1_final_y.append(cp1_y_sortdict[i][j])
                cp2_final_x.append(cp2_x_sortdict[i][j])
                cp2_final_y.append(cp2_y_sortdict[i][j])
        
        return np.concatenate(cp1_final_x), np.concatenate(cp1_final_y), np.concatenate(cp2_final_x), np.concatenate(
            cp2_final_y)
    
    def back_to_gait(self, x, y=None, avg_gait_samples=None):
        if avg_gait_samples is None:
            avg_gait_samples = self.cp1.avg_gait_samples
        x_return = np.reshape(x, (int(x.shape[0] / avg_gait_samples), avg_gait_samples, x.shape[1]), 'C')
        if y is not None:
            y_return = np.reshape(y, (int(y.shape[0] / avg_gait_samples), avg_gait_samples, y.shape[1]), 'C')
            return x_return, y_return
        
        return x_return
    
    def subsample(self, percent, cp1_x=None, cp1_y=None, cp2_x=None, cp2_y=None):
        # perhaps i am writing somethign that is 3 lines of code in 576 lines
        # but do i care?
        
        avg_gait_samples = self.cp1.avg_gait_samples
        
        if cp1_x is None:
            cp1_x = self.data['cp1']['proc_x']
        if cp1_y is None:
            cp1_y = self.data['cp1']['proc_y']
        if cp2_x is None:
            cp2_x = self.data['cp2']['proc_x']
        if cp2_y is None:
            cp2_y = self.data['cp2']['proc_y']
        
        if percent == 1.0:
            return cp1_x, cp1_y, cp2_x, cp2_y
        
        subsize = int(percent * cp1_x.shape[0])
        
        my_list = [cp1_x, cp1_y, cp2_x, cp2_y]
        new_array = []
        for array in my_list:
            new_array.append(array[:subsize, :])
        
        return new_array[0], new_array[1], new_array[2], new_array[3]
    
    def new_apply_ridge(self, x1=None, y1=None, x2=None, y2=None, metric_angle=None, decoder=None, my_alpha=100, k=10):
        if x1 is None:
            x1 = self.data['cp1']['proc_x']
        if y1 is None:
            y1 = self.data['cp1']['proc_y']
        if x2 is None:
            x2 = self.data['cp2']['proc_x']
        if y2 is None:
            y2 = self.data['cp2']['proc_y']
        if metric_angle is None:
            metric_angle = self.metric_angle
        
        angle_number = self.cp2.angle_name_helper(metric_angle)
        
        if decoder is None:
            b0, nada, nada, nada = self.cp1.decode_angles(scale=False)
        else:
            b0 = decoder
        
        kf = KFold(n_splits=k)
        best_vaf = -100000
        vaf_array = []
        best_h = None
        best_transformer = None
        best_predic = None
        best_y2_format = None
        for train_index, test_index in kf.split(x2):
            train_x2, test_x2 = x2[train_index, :], x2[test_index, :]
            train_y2, test_y2 = y2[train_index, :], y2[test_index, :]
            train_x1 = x1[train_index, :]
            
            transformer, train_x2_aligned = self.apply_CCA(cp1_x=train_x1, cp2_x=train_x1)
            
            train_x2_aligned_format, train_y2_format = format_data(train_x2_aligned, train_y2)
            
            new_h, _ = ridge_fit(b0, train_x2_aligned_format, train_y2_format, angle_number=angle_number)
            
            test_x2_aligned = self.quick_cca(test_x2, transformer, scale=False)
            
            test_x2_aligned_format, test_y2_format = format_data(test_x2_aligned, test_y2)
            test_predic = test_wiener_filter(test_x2_aligned_format, new_h)
            
            vaf_array.append(vaf(test_y2_format[:, angle_number], test_predic[:, angle_number]))
            print(f'test_set_score={vaf_array[-1]}')
            
            if vaf_array[-1] > best_vaf:
                best_vaf = vaf_array[-1]
                best_h = new_h
                best_transformer = transformer
                best_predic = test_predic
                best_y2_format = test_y2_format
        
        # x2_full = self.cp2.data['rates']
        # y2_full = self.cp2.data['angles']
        # x2_transform_full = []
        # for x in x2_full:
        #    x2_transform_full.append(self.quick_cca(x, transformer))
        
        # x2_scca_format, y2_format = self.cp2.stitch_and_format(x2_transform_full,
        #        y2_full)
        
        return best_transformer, best_h, vaf_array, best_predic, best_y2_format
    
    def old_apply_ridge(self, reduce_dims=False, dims=None, metric_angle=None, decoder=None, phase=True):
        if reduce_dims:
            if self.data['cp1']['pca_x'] is not None:
                x1 = self.data['cp1']['pca_x']
                x2 = self.data['cp2']['pca_x']
            else:
                x1, x2 = self.apply_PCA(preset_num_components=dims)
        else:
            x1 = self.data['cp1']['proc_x']
            x2 = self.data['cp2']['proc_x']
        
        if metric_angle is None:
            metric_angle = self.metric_angle
        
        angle_number = self.cp2.angle_name_helper(metric_angle)
        
        y1 = self.data['cp1']['proc_y']
        y2 = self.data['cp2']['proc_y']
        
        if decoder is None:
            if reduce_dims:
                temp_x_pca = self.cp1.apply_PCA(dims=x1.shape[1])
                b0, nada, nada, nada = self.cp1.decode_angles(X=temp_x_pca)
            else:
                b0, nada, nada, nada = self.cp1.decode_angles(scale=True)
        else:
            b0 = decoder
        transformer, nada = self.apply_CCA(cp1_x=x1, cp2_x=x2)
        
        x2_full = self.cp2.data['rates']
        y2_full = self.cp2.data['angles']
        x2_transform_full = []
        for x in x2_full:
            x2_transform_full.append(self.quick_cca(x, transformer))
        
        x2_scca_format, y2_format = self.cp2.stitch_and_format(x2_transform_full, y2_full)
        
        wpost, ywpost = ridge_fit(b0, x2_scca_format, y2_format, my_alpha=100, angle_number=angle_number)
        
        return transformer, wpost, ywpost
    
    def apply_ridge_only_proc(self, reduce_dims=False, dims=None, angle=6):
        '''deprecated I believe'''
        
        if reduce_dims:
            if self.data['cp1']['pca_x'] is not None:
                x1 = self.data['cp1']['pca_x']
                x2 = self.data['cp2']['pca_x']
            else:
                x1, x2 = self.apply_PCA(preset_num_components=dims)
        else:
            x1 = self.data['cp1']['proc_x']
            x2 = self.data['cp2']['proc_x']
        
        y1 = self.data['cp1']['proc_y']
        y2 = self.data['cp2']['proc_y']
        
        if reduce_dims:
            temp_x_pca = self.cp1.apply_PCA(dims=x1.shape[1])
            b0, nada, nada, nada = self.cp1.decode_angles(X=temp_x_pca)
        else:
            b0, nada, nada, nada = self.cp1.decode_angles(scale=True)
        transformer, x2_cca = self.apply_CCA(cp1_x=x1, cp2_x=x2)
        scaler = StandardScaler()
        x2_scca = scaler.fit_transform(x2_cca)
        
        print(x2_scca.shape, y2.shape)
        x2_scca_format, y2_format = format_data(x2_scca, y2)
        
        wpost, ywpost = ridge_fit(b0, x2_scca_format, y2_format, my_alpha=100, angle=angle)
        
        return transformer, wpost, ywpost
    
    def quick_cca(self, x, transformer, scale=True):
        nada, temp = transformer.transform(x, x)
        temp2 = transformer.inverse_transform(temp)
        if scale:
            scaler = StandardScaler()
            return scaler.fit_transform(temp2)
        else:
            return temp2
    
    def new_apply_pinv_transform(self, x=None, y=None, decoder=None, metric_angle='forelimb', k=10):
        if decoder is None:
            decoder, nada, nadax, naday = self.cp1.decode_angles()
        if x is None:
            x = self.data['cp2']['proc_x']
        if y is None:
            y = self.data['cp2']['proc_y']
        
        kf = KFold(n_splits=k)
        
        angle_number = self.cp2.angle_name_helper(metric_angle)
        vaf_array = []
        best_vaf = -100000
        best_transformer = None
        best_predic = None
        best_y_format = None
        for train_index, test_index in kf.split(x):
            train_x, test_x = x[train_index, :], x[test_index, :]
            train_y, test_y = y[train_index, :], y[test_index, :]
            
            train_x_format, train_y_format = format_data(train_x, train_y)
            
            transformer, _ = pinv_fit(decoder, train_x_format, train_y_format, angle=angle_number)
            
            test_x_format, test_y_format = format_data(test_x, test_y)
            
            predic = pinv_predicter(transformer, decoder, test_x_format)
            
            vaf_array.append(vaf(test_y_format[:, angle_number], predic[:, angle_number]))
            print(f'test_set_score={vaf_array[-1]}')
            
            if vaf_array[-1] > best_vaf:
                best_vaf = vaf_array[-1]
                best_transformer = transformer
                best_predic = predic
                best_y_format = test_y_format
        
        # x2_full = self.cp2.data['rates']
        # y2_full = self.cp2.data['angles']
        # x2_transform_full = []
        # for x in x2_full:
        #    x2_transform_full.append(self.quick_cca(x, transformer))
        
        # x2_scca_format, y2_format = self.cp2.stitch_and_format(x2_transform_full,
        #        y2_full)
        
        return best_transformer, vaf_array, best_predic, best_y_format
    
    def apply_pinv_transform(self, x=None, y=None, decoder=None):
        if decoder is None:
            decoder, nada, nadax, naday = self.cp1.decode_angles()
        if x is None:
            x = self.cp2.data['rates']
        if y is None:
            y = self.cp2.data['angles']
        
        if isinstance(x, list):
            x_format, y_format = self.cp2.stitch_and_format(x, y)
        else:
            x_format, y_format = format_data(x, y)
        clf, predic = pinv_fit(decoder, x_format, y_format)
        
        return clf, predic
    
    def remove_cp2_channels(self):
        samples = self.cp2.data['rates'][0].shape[0]
        channels = self.cp1.data['rates'][0].shape[1]
        new_cp2_rates = np.zeros((samples, channels))
        
        cp1_channels = self.cp1.data['which_channels']
        cp2_channels = self.cp2.data['which_channels']
        
        if len(cp2_channels) < len(cp1_channels):
            print('less channels, i dunno what to do, returning nothing')
            return
        else:
            i = 0
            for idx, channel in enumerate(cp2_channels):
                print(f'this is index:{idx}, and channel#:{channel}')
                if channel in cp1_channels:
                    new_cp2_rates[:, i] = self.cp2.data['rates'][0][:, idx]
                    print(f'idx={idx}, i={i}')
                    i = i + 1
                else:
                    print(f'removing channel {channel} from cp2')
        
        return new_cp2_rates
