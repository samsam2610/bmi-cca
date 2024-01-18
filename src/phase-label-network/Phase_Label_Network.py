# Credit: adopted codes originally written by Grant Engberson

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical, timeseries_dataset_from_array
from keras.preprocessing.image import ImageDataGenerator
import random as rand
from keras import backend as K
import keras
from keras.callbacks import LearningRateScheduler


class PhaseLabelNetwork:
    def __init__(self, x, y, N=100, sampling_rate=1, batch_size=1, con_window=24, pooling=2, max_epoch=100, shuffle=False, seed=None, start_index=None,
                 end_index=None, ):

        
        self.x = x
        self.y = y
        self.N = N
        self.con_window = con_window
        self.pooling = pooling
        self.max_epoch = max_epoch
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.start_index = start_index
        self.end_index = end_index
        
        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.callbacks = [LearningRateScheduler(self.lr_scheduler, verbose=1), EarlyStopping(monitor='val_loss', patience=5)]
        
        self.features, self.labels = None, None
        self.pre_train_indicies, self.test_indicies = None, None
        self.train_indicies, self.val_indicies = None, None
        self.X_test, self.y_test = None, None
        self.y_test_uncoded = None
        
        self.train_data, self.val_data = None, None
        self.model, self.history = None, None
    
    def timeseries_dataset_from_array_malleable(self, data, offset_targets, sequence_stride, N=None, sampling_rate=1, batch_size=1,
                                                shuffle=False, seed=None, start_index=None, end_index=None, ):
        """
        Creates a TensorFlow dataset from an array of time series data.

        Args:
            data (ndarray): The input time series data.
            offset_targets (ndarray): The target values for each time step.
            sequence_stride (int): The stride between consecutive sequences.
            N (int): The number of time steps in each sequence.
            sampling_rate (int, optional): The sampling rate of the time series data. Defaults to 1.
            batch_size (int, optional): The batch size for the dataset. Defaults to 1.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
            seed (int, optional): The random seed for shuffling the dataset. Defaults to None.
            start_index (int, optional): The starting index of the data to include in the dataset. Defaults to None.
            end_index (int, optional): The ending index of the data to include in the dataset. Defaults to None.

        Returns:
            tf.data.Dataset: The TensorFlow dataset containing the time series data and targets.
        """
        if N is None:
            N = self.N
            
        targets = np.asarray(np.roll(offset_targets, -N, axis=0), dtype="int64")
        sequence_length = (N * 2) + 1
        
        start_index = 0
        end_index = len(data)
        
        index_dtype = "int64"
        
        # Generate start positions
        start_positions = sequence_stride
        
        sequence_length = tf.cast(sequence_length, dtype=index_dtype)
        sampling_rate = tf.cast(sampling_rate, dtype=index_dtype)
        
        positions_ds = tf.data.Dataset.from_tensors(start_positions).repeat()
        
        indices = tf.data.Dataset.zip((tf.data.Dataset.range(len(start_positions)), positions_ds)).map(
            lambda i, positions: tf.range(positions[i], positions[i] + sequence_length * sampling_rate,
                                          sampling_rate, ), num_parallel_calls=tf.data.AUTOTUNE, )
        
        dataset = self.sequences_from_indices(data, indices, start_index, end_index)
        if targets is not None:
            indices = tf.data.Dataset.zip((tf.data.Dataset.range(len(start_positions)), positions_ds)).map(
                lambda i, positions: positions[i], num_parallel_calls=tf.data.AUTOTUNE, )
            target_ds = self.sequences_from_indices(targets, indices, start_index, end_index)
            dataset = tf.data.Dataset.zip((dataset, target_ds))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        if batch_size is not None:
            if shuffle:
                # Shuffle locally at each iteration
                dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
            dataset = dataset.batch(batch_size)
        else:
            if shuffle:
                dataset = dataset.shuffle(buffer_size=1024, seed=seed)
        return dataset
    
    def sequences_from_indices(self, array, indices_ds, start_index, end_index):
        """
        Creates a dataset of sequences from an array based on given indices.

        Args:
            array (ndarray): The input array.
            indices_ds (tf.data.Dataset): The dataset of indices.
            start_index (int): The starting index of the array.
            end_index (int): The ending index of the array.

        Returns:
            tf.data.Dataset: A dataset of sequences based on the given indices.
        """
        dataset = tf.data.Dataset.from_tensors(array[start_index:end_index])
        dataset = tf.data.Dataset.zip((dataset.repeat(), indices_ds)).map(lambda steps, inds: tf.gather(steps, inds),
                                                                          num_parallel_calls=tf.data.AUTOTUNE, )
        return dataset
    
    def format_test_data(self, x, y, start_indexes, N=None):
        if N is None:
            N = self.N
        
        longways = []
        y_indexes = start_indexes + N
        for i in start_indexes:
            sideways = x[i:i + N + N + 1, :]
            longways.append(sideways)
        y_formatted = y[y_indexes, :]
        return np.asarray(longways), y_formatted
    
    def crop_y(self, y, N=None):
        
        if N is None:
            N = self.N
            
        y_formatted = y[N:-N]
        return y_formatted
    
    def index_balancer(self, y, toggle="off"):
        """
        Resamples the input array `y` to balance the class distribution along the columns.

        Parameters:
            y (numpy.ndarray): Input array of shape (n_samples, n_classes) representing the class labels.
            toggle (str, optional): Toggle for adjusting the resampling likelihood. Default is "off".

        Returns:
            resampled_y (numpy.ndarray): Resampled array of shape (n_resampled_samples, n_classes).
            original_indicies (numpy.ndarray): Array of original indices corresponding to the resampled samples.
        """
        value_counts = []
        resampled_y = []
        original_indicies = []
        for i in range(y.shape[1]):
            column_sum = np.sum(y[:, i])
            value_counts.append(column_sum)
        print(value_counts)
        for i in range(y.shape[1]):
            floor = np.min(value_counts)
            extant = value_counts[i]
            if toggle == "on":
                likelihood = (extant - floor) / extant
            else:
                likelihood = 0
            for j in range(y.shape[0]):
                if y[j, i] == 1:
                    if rand.random() >= likelihood:
                        resampled_y.append(y[j, :])
                        original_indicies.append(j)
        return np.array(resampled_y), np.array(original_indicies)

    def reshaped_resampling(self, X, y):
        value_counts = []
        resampled_X = []
        resampled_y = []
        for i in range(y.shape[1]):
            column_sum = np.sum(y[:,i])
            value_counts.append(column_sum)
        for i in range(y.shape[1]):
            floor = np.min(value_counts)
            extant = value_counts[i]
            likelihood = (extant-floor)/extant
            for j in range(y.shape[0]):
                if y[j,i] == 1:
                    if rand.random() >= likelihood:
                        resampled_X.append(X[j, :, :])
                        resampled_y.append(y[j, :])
        return np.array(resampled_X), np.array(resampled_y)

    def lr_scheduler(self, epoch, lr):
        decay_rate = 0.9
        decay_step = 8
        if epoch % decay_step == 0 and epoch:
            return lr * pow(decay_rate, np.floor(epoch / decay_step))
        return lr

    def index_stacker(self, index_list_list, shapes):
        adj_ind = []
        for i in range(len(index_list_list)):
            adj = 0
            burner = i
            while burner >= 1:
                adj = adj + shapes[burner-1]
                burner = burner - 1
            appendage = index_list_list[i] + adj
            adj_ind = np.concatenate((adj_ind, appendage))
        return np.asarray(adj_ind, dtype=int)
    
    def f1(self, y_true, y_pred):
        y_pred = K.round(y_pred)
        tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
        tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2*p*r / (p+r+K.epsilon())
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
        return K.mean(f1)

    def f1_loss(self, y_true, y_pred):
        
        tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
        tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2*p*r / (p+r+K.epsilon())
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
        return 1 - K.mean(f1)

    def process_data(self, file_paths, N=None):
        if N is None:
            N = self.N
            
        # Initialize lists to store processed data
        all_X, all_y, all_y_cat_pref, all_bal_ind, shape_array = [], [], [], [], []

        # Process each file
        for path in file_paths:
            # Load data
            data = pd.read_csv(path, header=0)

            # Data allocation and scaling
            y = data.iloc[:, -1:]
            y_cat_pref = to_categorical(y)
            X = np.array(data.iloc[:, :-2])

            # Store features and labels
            all_X.append(X)
            all_y_cat_pref.append(y_cat_pref)
            shape_array.append(y_cat_pref.shape[0])

            # Index split tracking
            y_form = self.crop_y(y_cat_pref, N)
            y_bal, bal_ind = self.index_balancer(y_form)
            all_bal_ind.append(bal_ind)

        # Concatenate all features and labels
        self.features = np.vstack(all_X)
        self.labels = np.vstack(all_y_cat_pref)

        # Index Stacking and Train-Test Split
        indies = self.index_stacker([bal_ind for bal_ind in all_bal_ind], shape_array)
        self.pre_train_indicies, self.test_indicies = train_test_split(indies, train_size=0.9)
        self.train_indicies, self.val_indicies = train_test_split(self.pre_train_indicies, train_size=0.9)
        
        self.X_test, self.y_test = self.format_test_data(self.features, self.labels, self.test_indicies, N)
        self.y_test_uncoded = np.argmax(self.y_test, axis=1)

        self.train_data = self.timeseries_dataset_from_array_malleable(data=self.features, offset_targets=self.labels,
                                                                       sequence_stride=self.train_indicies, N=self.N,
                                                                       batch_size=1000)
        self.val_data = self.timeseries_dataset_from_array_malleable(data=self.features, offset_targets=self.labels,
                                                                        sequence_stride=self.val_indicies, N=self.N,
                                                                        batch_size=1000)
        
        return {
            'train_data': self.train_data,
            'val_data': self.val_data,
            'features': self.features,
            'labels': self.labels,
            'pre_train_indices': self.pre_train_indicies,
            'test_indices': self.test_indicies,
            'train_indices': self.train_indicies,
            'val_indices': self.val_indicies,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'y_test_uncoded': self.y_test_uncoded
        }

    def build_model(self, input_shape=None):
        if input_shape is None:
            if self.features is not None:
                input_shape = (self.N * 2 + 1, self.features.shape[1])
            else:
                raise Exception('input_shape is None. Please run process_data() before build_model()')

        model = Sequential()
        model.add(Conv1D(64, self.con_window, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(self.pooling))
        model.add(Dropout(0.3))
        model.add(Conv1D(128, self.con_window, activation='relu'))
        model.add(MaxPooling1D(self.pooling))
        model.add(Dropout(0.3))
        model.add(Conv1D(256, self.con_window, activation='relu'))
        model.add(MaxPooling1D(self.pooling))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dropout(0.3))
        model.add(Dense(100, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dropout(0.3))
        model.add(Dense(2, activation='sigmoid', kernel_initializer='glorot_uniform'))
        model.compile(loss=self.f1_loss, optimizer=self.optimizer, metrics=['accuracy', self.f1])
        
        self.model = model
        return model
    
    def fit_model(self, model=None, train_data=None, val_data=None):
        if train_data is None:
            train_data = self.train_data
        if val_data is None:
            val_data = self.val_data
        if model is None:
            model = self.model
            
        if self.train_data is None or self.val_data is None:
            raise Exception('train_data or val_data is None. Please run process_data() before fit_model()')
        if self.model is None:
            raise Exception('model is None. Please run build_model() before fit_model()')
        
        history = model.fit(train_data, epochs=self.max_epoch, validation_data=val_data, callbacks=self.callbacks)
        
        self.history = history
        model.save('model.h5')
        return history