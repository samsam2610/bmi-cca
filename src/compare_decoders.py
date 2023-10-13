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


class DecodersComparison:
    def __init__(self, cp1_index, cp2_index, subsample_list, path=None, data_dict=None, sorted_keys=None, step=0.005,
                 pca_dims=8, split_ratio=0.8, num_processes=6, multiThread=False, multiProcess=False):
        
        self.pca_dims = pca_dims
        self.split_ratio = split_ratio
        self.subsample_list = subsample_list
        self.step = step
        
        self.cp1 = None
        self.cp2 = None
        self.cp2_original = None
        self.cp2_test = None
        
        # Load data
        # If path is specified, load data from folder and ignore data_dict and sorted_keys
        if path is not None:
            self.data_dict, self.sorted_keys = load_data_from_folder(path=path)
            self.assign_data(cp1_index=cp1_index, cp2_index=cp2_index)
            print(f"Loaded data from {path}")
        else:
            if data_dict is None or sorted_keys is None:
                raise Exception("data_dict and sorted_keys must be specified if path is not specified")
            
            self.data_dict = data_dict
            self.sorted_keys = sorted_keys
            self.assign_data(cp1_index=cp1_index, cp2_index=cp2_index)
            print(f"Loaded data from data_dict and sorted_keys")
        
        if self.cp1 is None or self.cp2 is None or self.cp2_test is None:
            raise Exception("cp1, cp2, and cp2_test must be assigned before running the decoder comparison")
        else:
            # Day-0 decoder stuffs
            self.day0_decoder, self.day0_transformer, self.day0_decoder_no_offset, self.offset, self.day0_decoder_scale = self.get_day0_decoder()
            self.subsample_subgroups, self.subsample_subgroups_index = split_range_into_groups(subsample_list,
                                                                                               num_processes)
       
        # Split data
        # Train-Test split
        self.percent_data = self.split_data()
        
        # Score keepers
        self.fixed_decoder_scores = None
        self.pca_decoder_scores = None
        self.cca_decoder_scores = None
        self.r_scores = None
        self.pinv_scores = None
    
        print("Finished initializing DecodersComparison object - Test4")
        
    def load_data(self, path):
        self.data_dict, self.sorted_keys = load_data_from_folder(path=path)
    
    def assign_data(self, cp1_index, cp2_index):
        self.cp1 = self.data_dict[self.sorted_keys[cp1_index]]
        self.cp2_original = self.data_dict[self.sorted_keys[cp2_index]]
        
        self.cp2 = copy.deepcopy(self.cp2_original)
        self.cp2_test = copy.deepcopy(self.cp2_original)
        print('Data assigned!')
    
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
        step = self.step
        pca_dims = self.pca_dims
        day0_transformer = self.day0_transformer
        day0_decoder = self.day0_decoder
        day0_decoder_no_offset = self.day0_decoder_no_offset
        day0_decoder_scale = self.day0_decoder_scale
        offset = self.offset
        
        # Train-Test split
        cp2 = copy.deepcopy(self.cp2)
        cp2_test = copy.deepcopy(self.cp2_test)
        cp1 = copy.deepcopy(self.cp1)
        percent_data = self.percent_data
        
        # prepare copy of data
        cp2_test_raw = copy.deepcopy(cp2_test)
        cp2_raw = copy.deepcopy(cp2)
        
        describe_dict(cp1.data)
        print()
        describe_dict(cp2.data)
        print()
        describe_dict(cp2_test.data)
        
        # get day0 decoder weights, scaled + unscaled versions
        # day0_decoder, day0_transformer, day0_decoder_no_offset, offset, day0_decoder_scale = self.get_day0_decoder()
        
        temp_cca = CCAProcessor(cp1, cp2)
        
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
    
    def compare_decoders(self, multiThread=False, multiProcess=False, num_processes=6):
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
        
        pca_decoder_scores, cca_decoder_scores, pinv_scores, r_scores, number_of_gaits, fixed_decoder_scores = {}, {}, {}, {}, {}, {}
        
        if multiThread is True:
            results = []
            # Create a ThreadPoolExecutor with the desired number of processes
            executor = ThreadPoolExecutor()
            # Submit tasks to the executor
            futures = [executor.submit(self.process_subsample_data, index_group) for index_group in
                       range(num_processes)]
            # Wait for all tasks to complete and retrieve the results
            for future in as_completed(futures):
                results.append(future.result())
        
        elif multiProcess is True:
            # Create a ProcessPoolExecutor with the desired number of processes
            executor = ProcessPoolExecutor(max_workers=num_processes)
            # Submit tasks to the executor
            futures = [executor.submit(self.process_subsample_data, index_group) for index_group in
                       range(num_processes)]
            
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
        
        # Create a dictionary to store the scores
        scores_dict = {'cca_decoder_scores': cca_decoder_scores, 'pca_decoder_scores': pca_decoder_scores,
                       'fixed_decoder_scores': fixed_decoder_scores, 'r_scores': r_scores, 'pinv_scores': pinv_scores,
                       'number_of_gaits': number_of_gaits}
        
        # Store the scores
        self.cca_decoder_scores = cca_decoder_scores
        self.pca_decoder_scores = pca_decoder_scores
        self.fixed_decoder_scores = fixed_decoder_scores
        self.r_scores = r_scores
        self.pinv_scores = pinv_scores
        
        return scores_dict
    
    def plot_vaf_comparison_multiple(self, number_of_gaits, title_str, path, max_gait=65, min_gait=0, ylim=None):
        """
        Plots the VAF (Variance Accounted For) comparison for different decoders.

        Parameters:
        number_of_gaits (int): The number of gaits to plot.
        title_str (str): The title string for the plot.
        path (str): The path to save the plot.
        max_gait (int, optional): The maximum gait number to include. Defaults to 65.
        ylim (tuple, optional): The limits for the y-axis as a tuple (bottom, top). Defaults to None.
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
        fig.set_size_inches(15, 5)
        ax1 = fig.add_subplot(111)
        x = self.subsample_list
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
        for i in range(len(self.subsample_list)):
            num_gaits.append(int(np.average(number_of_gaits[i])))
            if self.pca_decoder_scores is not None:
                pca_scores_avg.append(np.average(self.pca_decoder_scores[i]))
                pca_scores_std.append(np.std(self.pca_decoder_scores[i]))
            
            if self.cca_decoder_scores is not None:
                cca_scores_avg.append(np.average(self.cca_decoder_scores[i]))
                cca_scores_std.append(np.std(self.cca_decoder_scores[i]))
            
            if self.r_scores is not None:
                r_scores_avg.append(np.average(self.r_scores[i]))
                r_scores_std.append(np.std(self.r_scores[i]))
            
            if self.pinv_scores is not None:
                pinv_scores_avg.append(np.average(self.pinv_scores[i]))
                pinv_scores_std.append(np.std(self.pinv_scores[i]))
            
            if self.fixed_decoder_scores is not None:
                if isinstance(self.fixed_decoder_scores, dict):
                    fixed_decoder_score_clean.append(self.fixed_decoder_scores[i][0])
                else:
                    fixed_decoder_score_clean = self.fixed_decoder_scores
        
        if self.fixed_decoder_scores is not None:
            ax1.plot(num_gaits, fixed_decoder_score_clean, '--', color='tab:pink', label='fixed_decoder')
        if self.pca_decoder_scores is not None:
            peak = np.full(len(num_gaits), max(pca_scores_avg))
            ax1.errorbar(num_gaits, pca_scores_avg, yerr=pca_scores_std, marker='o', color='tab:blue',
                         label='pca decoder')
            ax1.plot(num_gaits, peak, '--', color='tab:blue', label='max pca decoder')
        if self.cca_decoder_scores is not None:
            ax1.errorbar(num_gaits, cca_scores_avg, yerr=cca_scores_std, marker='s', color='tab:green',
                         label='cca aligned')
        if self.r_scores is not None:
            ax1.errorbar(num_gaits, r_scores_avg, yerr=r_scores_std, marker='*', color='tab:orange',
                         label='preloaded decoder')
        if self.pinv_scores is not None:
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
