import os


class FolderHandler:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.angles_list = self.get_angles_list()
        self.coords_list = self.get_coords_list()
        self.tdt_list, self.ts_list = self.get_tdt_ts_paths()  # TODO add checks
    
    def get_tdt_ts_paths(self):
        tdt_path = f'{self.folder_path}/tdt'
        
        unsorted = []
        time_list = []
        ts_path_list = []
        ignore_extensions = ('.pdf', '.DS_Store')
        
        for file_name in os.listdir(tdt_path):
            if file_name.endswith(ignore_extensions):
                continue  # ignore this file
            time_list.append(file_name.split('-')[-1])
            unsorted.append(f'{tdt_path}/{file_name}')
        # canned code to sort by attempt #
        tdt_path_list = [unsorted for _, unsorted in sorted(zip(time_list, unsorted))]
        
        for temp in tdt_path_list:
            ts_path = f'{temp}/live_videos'
            for file_name in os.listdir(ts_path):
                if file_name.endswith('.npy'):
                    ts_path_list.append(f'{ts_path}/{file_name}')
                    break
        
        return tdt_path_list, ts_path_list
    
    def deprec_get_tdt_list(self):
        
        tdt_path = f'{self.folder_path}/tdt'
        
        unsorted = []
        attempt_num_list = []
        
        for file_name in os.listdir(tdt_path):
            subject_name = file_name.split('-')[0]
            attempt_num = int(subject_name.split('_')[1])
            
            unsorted.append(f'{tdt_path}/{file_name}')
            attempt_num_list.append(attempt_num)
        
        # canned code to sort by attempt #
        tdt_path_list = [unsorted for _, unsorted in sorted(zip(attempt_num_list, unsorted))]
        
        return tdt_path_list
    
    def get_angles_list(self):
        angles_path = f'{self.folder_path}/angles'
        unsorted = []
        time_list = []
        
        for file_name in os.listdir(angles_path):
            if file_name.endswith('.csv'):
                subject_name = file_name.split('_')[1]
                time_list.append(subject_name.split('-')[-1])
                unsorted.append(f'{angles_path}/{file_name}')
        
        angles_path_list = [unsorted for _, unsorted in sorted(zip(time_list, unsorted))]
        
        return angles_path_list
    
    def get_coords_list(self):
        coords_path = f'{self.folder_path}/pose-3d'
        unsorted = []
        time_list = []
        
        for file_name in os.listdir(coords_path):
            if file_name.endswith('.csv'):
                subject_name = file_name.split('_')[1]
                time_list.append(subject_name.split('-')[-1])
                unsorted.append(f'{coords_path}/{file_name}')
        
        coords_path_list = [unsorted for _, unsorted in sorted(zip(time_list, unsorted))]
        
        return coords_path_list
    
    def deprec_get_ts_list(self):
        
        vids_path = f'{self.folder_path}/vids'
        
        unsorted = []
        attempt_num_list = []
        substring = 'cam1'
        
        for file_name in os.listdir(vids_path):
            if file_name.endswith('.npy'):
                if substring in file_name:
                    subject_name = file_name.split('-')[0]
                    attempt_num = int(subject_name.split('_')[4])
                    
                    unsorted.append(f'{vids_path}/{file_name}')
                    attempt_num_list.append(attempt_num)
        
        ts_path_list = [unsorted for _, unsorted in sorted(zip(attempt_num_list, unsorted))]
        
        return ts_path_list
