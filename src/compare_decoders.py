import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils import *
from src.wiener_filter import *


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


class DecodersComparison:
    def __init__(self, cp1, cp2, subsample_list, step=0.005, pca_dims=8, split_ratio=0.8, num_processes=6,
                 multiThread=False, multiProcess=False):
        
        self.cp1 = cp1
        
        self.cp2_original = cp2
        self.cp2 = copy.deepcopy(self.cp2_original)
        self.cp2_test = copy.deepcopy(self.cp2_original)
        
        self.pca_dims = pca_dims
        self.split_ratio = split_ratio
        self.subsample_list = subsample_list
        self.step = step
        
        # Day-0 decoder stuffs
        self.day0_decoder, self.day0_transformer, self.day0_decoder_no_offset, self.offset, self.day0_decoder_scale = self.get_day0_decoder()
        self.subsample_subgroups, self.subsample_subgroups_index = split_range_into_groups(subsample_list,
                                                                                           num_processes)
    
    # Function to split the data into train and test sets
    def split_data(self):
        # split only the cp2 data
        cp2_length = self.cp2.data['rates'][0].shape[0]
        percent_data = int(self.split_ratio * cp2_length)
        
        self.cp2.data['rates'] = [self.cp2.data['rates'][0][:percent_data, :]]
        self.cp2.data['angles'] = [self.cp2.data['angles'][0][:percent_data, :]]
        
        self.cp2_test.data['rates'] = [self.cp2_test.data['rates'][0][percent_data:, :]]
        self.cp2_test.data['angles'] = [self.cp2_test.data['angles'][0][percent_data:, :]]
        
        return percent_data
    
    # Function to obtain the day0 decoder weights
    def get_day0_decoder(self):
        X_tempy = self.cp1.apply_PCA(dims=self.pca_dims)
        scaler = StandardScaler()
        X_scale = scaler.fit_transform(np.squeeze(X_tempy))
        day0_decoder_scale, _, _, _ = self.cp1.decode_angles(X=[X_scale])
        
        day0_decoder, _, _, _ = self.cp1.decode_angles(X=self.cp1.apply_PCA(dims=self.pca_dims))
        day0_transformer = self.cp1.pca_object
        day0_decoder_no_offset = day0_decoder[1:, :]
        offset = day0_decoder[0, :]
        
        return day0_decoder, day0_transformer, day0_decoder_no_offset, offset, day0_decoder_scale
    
    # Get PCA decoder score
    def get_pca_decoder_score(self, sub_x, sub_y, test_y):
        try:
            # apply PCA just using sub-sampled data on the rates (x) data
            cp2_test = copy.deepcopy(self.cp2_test)
            cp2 = copy.deepcopy(self.cp2)
            
            sub_x_pca = cp2.apply_PCA(dims=self.pca_dims, X=sub_x)  # don't np.squeeze this line
            
            # save PCA transformation
            pca_object = cp2.pca_object
            
            # train PCA decoder
            temp_h, _, _, _ = cp2.decode_angles(X=sub_x_pca, Y=sub_y)
            
            # test PCA decoder
            test_x_pca = np.squeeze(np.array(cp2_test.apply_PCA(dims=self.pca_dims, transformer=pca_object)))
            test_x_pca_format, test_y_format = format_data(test_x_pca, test_y)
            
            temp_y = test_wiener_filter(test_x_pca_format, temp_h)
            
            vaf_score = vaf(test_y_format[:, 1], temp_y[:, 1])
            return vaf_score, None
        except Exception as e:
            print(f"Traceback: \n {traceback.format_exc()}")
            return None, e
    
    # Get CCA decoder score
    def get_cca_decoder_score(self, sub_x, sub_y):
        try:
            # Prepare the CCA object by aligning Day-N subsampled data to Day-0 data by fitting PCA
            # make subsampled cp2
            cp2_test = copy.deepcopy(self.cp2_test)
            cp2 = copy.deepcopy(self.cp2)
            cp1 = copy.deepcopy(self.cp1)
            
            cp2.data['rates'] = sub_x
            cp2.data['angles'] = sub_y
            
            temp_cca = CCAProcessor(cp1, cp2)
            # check pca_dims validity. Slicing the data too small -> not enough data for features
            check_pca_dims = min(len(temp_cca.data['cp2']['proc_x']), len(temp_cca.data['cp1']['proc_x']),
                                 self.pca_dims)
            if check_pca_dims >= self.pca_dims:
                # fit Day-0 PCA transformation on subsampled Day-n data, and then transform it
                pca_sub_x1, pca_sub_x2 = temp_cca.apply_PCA(preset_num_components=self.pca_dims,
                                                            day_0_transformer=self.day0_transformer)
            
            # fit a CCA transformation on subsampled, low-D day-N data
            temp_cca_transformer, _ = temp_cca.apply_CCA(preset_num_components=self.pca_dims, pca=True)
        
        except Exception as e:
            print(f"Traceback: \n {traceback.format_exc()}")
            return None, e, None, None
        
        try:
            # test CCA
            # apply our PCA transformation to entire Day-N,
            # and then apply our CCA transformation to entire low-D day-N data
            test_x_squeeze = np.squeeze(
                np.array(cp2_test.apply_PCA(dims=self.pca_dims, transformer=temp_cca.data['cp2']['pca_transformer'])))
            test_y_squeeze = np.squeeze(np.array(cp2_test.data['angles']))
            
            # temp_x_cca_space [=] sub-sampled, low-d, CCA-transformed day-n data rates
            _, test_x_cca_space = temp_cca_transformer.transform(test_x_squeeze, test_x_squeeze)
            
            # then transform back to the original space
            temp_x = temp_cca_transformer.inverse_transform(test_x_cca_space)
            temp_x_format, test_y_format = format_data(temp_x, test_y_squeeze)
            
            if check_pca_dims >= self.pca_dims:
                temp_y = test_wiener_filter(temp_x_format, self.day0_decoder)
            vaf_score = vaf(test_y_format[:, 1], temp_y[:, 1])
            return vaf_score, None, temp_cca, temp_cca_transformer
        
        except Exception as e:
            print(f"Traceback: \n {traceback.format_exc()}")
            return None, e, temp_cca, temp_cca_transformer
    
    # Get regression fit score
    def get_regression_fit_score(self, temp_cca, temp_cca_transformer, sub_x, sub_y):
        try:
            # sub_x_pca [=] subsampled rates data in PCA space
            # apply PCA to subsampled data
            cp2_test = copy.deepcopy(self.cp2_test)
            cp2 = copy.deepcopy(self.cp2)
            sub_x_pca = cp2.apply_PCA(dims=self.pca_dims, X=sub_x)  # don't np.squeeze this line
            sub_x_pca_squeezed = np.squeeze(sub_x_pca)
            
            # transform PCA subsample to day-0 shape
            _, sub_x_cca_space = temp_cca_transformer.transform(sub_x_pca_squeezed, sub_x_pca_squeezed)
            
            # then transform back to the original space
            temp_x = temp_cca_transformer.inverse_transform(sub_x_cca_space)
            
            # scale transformed data
            scaler = StandardScaler()
            temp_x_scale = scaler.fit_transform(temp_x)
            # temp_x_scale = scaler.transform(temp_x)
            temp_x_format, temp_y_format = format_data(temp_x_scale, sub_y[0])
            
            # regression fit the scaled data
            wpost, _ = ridge_fit(b0=self.day0_decoder_scale, x_format=temp_x_format, y_format=temp_y_format,
                                 my_alpha=100.0)
            
            # preloaded decoder
            test_x_squeeze = np.squeeze(np.array(cp2_test.apply_PCA(dims=self.pca_dims, transformer=cp2.pca_object)))
            test_y_squeeze = np.squeeze(np.array(cp2_test.data['angles']))
            _, test_x2tox1 = temp_cca.apply_CCA(cp2_x=test_x_squeeze, transformer=temp_cca_transformer)
            test_x2tox1_scale = scaler.transform(test_x2tox1)
            test_x2tox1_scale_format, test_y_format = format_data(test_x2tox1_scale, test_y_squeeze)
            temp_y = test_wiener_filter(test_x2tox1_scale_format, wpost)
            vaf_score = vaf(test_y_format[:, 1], temp_y[:, 1])
            return vaf_score, None
        except Exception as e:
            print(f"Traceback: \n {traceback.format_exc()}")
            return None, e
    
    # Function to obtain number of gait in the subsampled data
    def get_gait_count(self, sub_y):
        temp_gaits, _ = self.cp2.get_gait_indices(Y=sub_y)
        num_gaits = temp_gaits[0].size - 1
        return num_gaits
    
    # def process_subsample_data(cp1, cp2_raw, cp2_test_raw, test_y, pca_dims, day0_transformer, day0_decoder, day0_decoder_no_offset, day0_decoder_scale, temp_cca, offset, scaler, index_group, subsample_subgroups, subsample_subgroups_index, percent_data, subsample_list, step, pca_decoder_scores, cca_decoder_scores, pinv_scores, r_scores, number_of_gaits):
    def process_subsample_data(self, index_group):
        cp1 = copy.deepcopy(self.cp1)
        step = self.step
        pca_dims = self.pca_dims
        
        # Train-Test split
        cp2, cp2_test, percent_data = self.split_data()
        
        # prepare copy of data
        cp2_test_raw = copy.deepcopy(cp2_test)
        cp2_raw = copy.deepcopy(cp2)
        
        describe_dict(cp1.data)
        print()
        describe_dict(cp2.data)
        print()
        describe_dict(cp2_test.data)
        
        # get day0 decoder weights, scaled + unscaled versions
        day0_decoder, day0_transformer, day0_decoder_no_offset, offset, day0_decoder_scale = self.get_day0_decoder()
        
        temp_cca = CCAProcessor(cp1, cp2)
        # scores keeper
        
        # transformer keeper
        # pca_predic, number_of_gaits, cca_transformers, r_predic, pinv_predic_all = {}, {}, {}, {}, {}
        
        # prepare test y data
        test_y = np.squeeze(np.array(cp2_test.data['angles']))
        pca_decoder_scores, cca_decoder_scores, pinv_scores, r_scores, number_of_gaits, fixed_decoder_scores = {}, {}, {}, {}, {}, {}
        subsample_list = self.subsample_subgroups[index_group]
        subsample_list_index = self.subsample_subgroups_index[index_group]
        
        for i in range(len(subsample_list)):
            cp2 = copy.deepcopy(cp2_raw)
            cp2_test = copy.deepcopy(cp2_test_raw)
            
            sub_idx = subsample_list_index[i]
            print(f"current idx is {sub_idx}")
            # slicing prepare
            slice_sample = int(subsample_list[i] * percent_data)
            step_sample = int(step * percent_data)
            
            # preallocate score keepers
            pca_decoder_scores[sub_idx], cca_decoder_scores[sub_idx], pinv_scores[sub_idx], r_scores[sub_idx], \
                number_of_gaits[sub_idx], fixed_decoder_scores[sub_idx] = [], [], [], [], [], []
            
            # testing decoder
            # day-0 fixed decoder
            try:
                test_x_fixed = np.squeeze(np.array(cp2_test.apply_PCA(dims=pca_dims, transformer=day0_transformer)))
                test_x_fixed_format, test_y_format = format_data(test_x_fixed, test_y)
                
                temp_y = test_wiener_filter(test_x_fixed_format, day0_decoder)
                fixed_decoder_scores[sub_idx].append(vaf(test_y_format[:, 1], temp_y[:, 1]))
                print(f" Succeeded fixed decoder - slice #: {slice_sample}, step #: {step_sample}")
            except Exception as e:
                print(f" Failed fixed decoder - {e} - slice #: {slice_sample}, step #: {step_sample}\n")
                print(f"Traceback: \n {traceback.format_exc()}")
            
            max_idx = len(np.arange(0, percent_data - step_sample, step_sample))
            for idx, k in enumerate(np.arange(0, percent_data - step_sample, step_sample)):
                cp2 = copy.deepcopy(cp2_raw)
                end_slice = int(k + slice_sample)
                print(
                    f" Current subsample#: {subsample_list[i]}, subsample index #: {(i + 1)}/{len(subsample_list)}, group #: {index_group}, step #: {idx}/{max_idx},  slice #: {slice_sample}, step #: {step_sample}, start: {k}, end: {end_slice}")
                if k + slice_sample > percent_data:
                    break
                sub_x = [cp2.data['rates'][0][int(k):end_slice, :]]
                sub_y = [cp2.data['angles'][0][int(k):end_slice, :]]
                
                # Process sub_x and sub_y
                num_gaits = self.get_gait_count(sub_y=sub_y)
                number_of_gaits[sub_idx].append(num_gaits)
                
                # Get PCA decoder score
                vaf_score, error_message = self.get_pca_decoder_score(sub_x=sub_x, sub_y=sub_y, test_y=test_y)
                
                if vaf_score is not None:
                    pca_decoder_scores[sub_idx].append(vaf_score)
                    print(
                        f" Succeeded pca decoder - gaits #: {num_gaits}, subsample index #: {(i + 1)}/{len(subsample_list)}, group #: {index_group}, step #: {idx}/{max_idx}, slice #: {slice_sample}, step #: {step_sample}, start: {k}, end: {end_slice}")
                else:
                    print(
                        f" Failed pca decoder - {error_message} - gaits #: {num_gaits}, {(i + 1) / len(subsample_list)}, group #: {index_group}, slice #: {slice_sample}, step #: {step_sample}, start: {k}, end: {end_slice}\n")
                
                # Get CCA decoder score
                try:
                    vaf_score, error_message, temp_cca, temp_cca_transformer = self.get_cca_decoder_score(sub_x=sub_x,
                                                                                                          sub_y=sub_y)
                    if vaf_score is not None:
                        cca_decoder_scores[sub_idx].append(vaf_score)
                        print(
                            f" Succeeded cca decoder - gaits #: {num_gaits}, subsample index #: {(i + 1)}/{len(subsample_list)}, group #: {index_group}, step #: {idx}/{max_idx}, slice #: {slice_sample}, step #: {step_sample}, start: {k}, end: {end_slice}")
                    else:
                        print(
                            f" Failed cca decoder - {error_message} - gaits #: {num_gaits}, {(i + 1) / len(subsample_list)}, group #: {index_group}, slice #: {slice_sample}, step #: {step_sample}, start: {k}, end: {end_slice}\n")
                except Exception as e:
                    print(
                        f" Failed cca decoder - {e} - gaits #: {num_gaits}, subsample index #: {(i + 1)}/{len(subsample_list)}, group #: {index_group}, step #: {idx}/{max_idx}, slice #: {slice_sample}, step #: {step_sample}, start: {k}, end: {end_slice}\n")
                    print(f"Traceback: \n {traceback.format_exc()}")
                
                # Regression fit
                try:
                    vaf_score, error_message = self.get_regression_fit_score(temp_cca=temp_cca,
                                                                             temp_cca_transformer=temp_cca_transformer,
                                                                             sub_x=sub_x, sub_y=sub_y)
                    if vaf_score is not None:
                        r_scores[sub_idx].append(vaf_score)
                    else:
                        print(
                            f" Failed regression fit - {error_message} - gaits #: {num_gaits}, {(i + 1) / len(subsample_list)}, group #: {index_group}, slice #: {slice_sample}, step #: {step_sample}, start: {k}, end: {end_slice}\n")
                except Exception as e:
                    print(
                        f" Failed regression fit - {e} - gaits #: {num_gaits}, subsample index #: {(i + 1)}/{len(subsample_list)}, group #: {index_group}, step #: {idx}/{max_idx}, slice #: {slice_sample}, step #: {step_sample}, start: {k}, end: {end_slice}\n")
                    print(f"Traceback: \n {traceback.format_exc()}")
                
                # Pinv fit
                try:
                    cp2_test = copy.deepcopy(cp2_test_raw)
                    pinv_clf, _ = temp_cca.apply_pinv_transform(x=sub_x[0], y=sub_y[0], decoder=day0_decoder)
                    
                    # test
                    raw_test_x = cp2_test.data['rates'][0]
                    raw_test_y = cp2_test.data['angles'][0]
                    raw_test_x_format, raw_test_y_format = format_data(raw_test_x, raw_test_y)
                    trans_test_x = pinv_clf.predict(raw_test_x_format)
                    pinv_predict = np.dot(trans_test_x, day0_decoder_no_offset) + offset
                    
                    pinv_predic_all = pinv_predict
                    pinv_scores[sub_idx].append(vaf(test_y_format[:, 1], pinv_predict[:, 1]))
                    print(
                        f" Succeeded pinv fit - gaits #: {num_gaits}, subsample index #: {(i + 1)}/{len(subsample_list)}, group #: {index_group}, step #: {idx}/{max_idx}, slice #: {slice_sample}, step #: {step_sample}, start: {k}, end: {end_slice}\n")
                except Exception as e:
                    print(
                        f" Failed pinv fit - {e} - gaits #: {num_gaits}, subsample index #: {(i + 1)}/{len(subsample_list)}, group #: {index_group}, step #: {idx}/{max_idx}, slice #: {slice_sample}, step #: {step_sample}, start: {k}, end: {end_slice}\n")
                    print(f"Traceback: \n {traceback.format_exc()}")
        
        scores_dict = {'fixed_decoder_scores': fixed_decoder_scores, 'pca_decoder_scores': pca_decoder_scores,
                       'cca_decoder_scores': cca_decoder_scores, 'r_scores': r_scores, 'pinv_scores': pinv_scores,
                       'number_of_gaits': number_of_gaits}
        
        return scores_dict
    
    def compare_decoders(self, subsample_list, multiThread=False, multiProcess=False, num_processes=6, step=0.005,
                         pca_dims=8, split_ratio=0.8):
        """
        Compares the performance of different decoders on the given CortProcessor objects.

        Parameters:
        cp1 (CortProcessor): The first CortProcessor object.
        cp2 (CortProcessor): The second CortProcessor object.
        subsample_list (list): A list of subsample sizes to use for evaluation.
        pca_dims (int, optional): The number of dimensions to use for PCA. Defaults to 8.
        split_ratio (float, optional): The ratio to split the data into training and testing sets. Defaults to 0.8.

        Returns:
        scores_dict: a dictionary contained the result of the following decoders
            
               cca_decoder_scores (list): The scores of CCA decoder for each subsample size.
               pca_decoder_scores (list): The scores of PCA decoder for each subsample size.
               fixed_decoder_scores (list): The scores of fixed decoder for each subsample size.

        Examples:
        scores_dict = compare_decoders(cp1, cp2,
                                        subsample_list,
                                        pca_dims=10,
                                        split_ratio=0.8)
        """
        # Train-Test split
        cp2_raw = copy.deepcopy(self.cp2_original)
        cp1_raw = copy.deepcopy(self.cp1)
        
        pca_decoder_scores, cca_decoder_scores, pinv_scores, r_scores, number_of_gaits, fixed_decoder_scores = {}, {}, {}, {}, {}, {}
        
        if multiThread is True:
            results = []
            # Create a ThreadPoolExecutor with the desired number of processes
            executor = ThreadPoolExecutor()
            # Submit tasks to the executor
            futures = [executor.submit(self.process_subsample_data, index_group) for
                       index_group in range(num_processes)]
            # Wait for all tasks to complete and retrieve the results
            for future in as_completed(futures):
                results.append(future.result())
        
        elif multiProcess is True:
            # Create a ProcessPoolExecutor with the desired number of processes
            executor = ProcessPoolExecutor(max_workers=num_processes)
            # Submit tasks to the executor
            futures = [executor.submit(self.process_subsample_data, index_group) for
                       index_group in range(num_processes)]
            
            # Wait for all tasks to complete and retrieve the results
            results = [future.result() for future in futures]
        else:
            results = []
            for index_group in range(num_processes):
                result = self.process_subsample_data(index_group)
                results.append(result)
        
        for result in results:
            if result is not None:
                score_dict = result
                pca_decoder_scores = {**pca_decoder_scores, **score_dict['pca_decoder_scores']}
                cca_decoder_scores = {**cca_decoder_scores, **score_dict['cca_decoder_scores']}
                fixed_decoder_scores = {**fixed_decoder_scores, **score_dict['fixed_decoder_scores']}
                r_scores = {**r_scores, **score_dict['r_scores']}
                pinv_scores = {**pinv_scores, **score_dict['pinv_scores']}
                number_of_gaits = {**number_of_gaits, **score_dict['number_of_gaits']}
        
        # # Create a dictionary to store the scores
        scores_dict = {'cca_decoder_scores': cca_decoder_scores, 'pca_decoder_scores': pca_decoder_scores,
                       'fixed_decoder_scores': fixed_decoder_scores, 'r_scores': r_scores, 'pinv_scores': pinv_scores,
                       'number_of_gaits': number_of_gaits}
        
        return scores_dict
