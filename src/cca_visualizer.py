from src.neural_analysis import *
from src.cort_processor import *
from src.wiener_filter import *
from src.folder_handler import *
from src.tdt_support import *
from src.decoders import *

from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold
from src.cca_processor import CCAProcessor
from datetime import datetime
import matplotlib.pyplot as plt
import itertools


class CCAVisualizer:
    def __init__(self, cca_object: CCAProcessor, saving_path):
        self.cca_object = cca_object
        self.saving_path = saving_path
    
    def plot_cca_gaits(self, offset_value=180, start_idx=0, end_idx=3000, title_string=None):
        # Create a plot
        plt.figure(figsize=(40, 5))
       
        cp1_data = [row[2] for row in self.cca_object.data['cp1']['proc_y']]
        cp2_data = [row[2] for row in self.cca_object.data['cp2']['proc_y']]
        dataset = [cp1_data, cp2_data]
        offset_values = [0, offset_value]
        names = ['CP1', 'CP2']
        colors = itertools.cycle(['red', 'green', 'blue'])
        
        for index, data in enumerate(dataset):
            # Define a cycle of three colors

            offset_value = offset_values[index]
            # Flatten the data into a single list and prepare indices
            indices = range(len(data))
            
            # Plot each sublist as a segment of a continuous line
            plt.plot(indices[start_idx:end_idx], data[start_idx:end_idx], color=next(colors))
        
        if title_string is None:
            title_string = 'Selected Gaits of CP1 and CP2'
            
        # Adding labels and title
        plt.xlabel('Overall Index')
        plt.ylabel('Angle (degrees)')
        plt.title(title_string)
        
        current_datetime = datetime.now().strftime('%Y%m%d-%H%M')
        pdf_file_path = os.path.join(self.saving_path, f'{title_string}_{current_datetime}.pdf')
        plt.savefig(pdf_file_path)
        # Display the plot
        plt.show()