import torch
import torch.nn as nn
from math import sqrt

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class Dataset_MIMIC(Dataset):
    def __init__(self, root_path, flag='train', max_len=8,
                 features='S', data_path='MIMICtable_261219.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.max_len = max_len

        self.root_path = root_path
        self.data_path = data_path
        self.tot_len = 0
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        # Define and Seperate Columns
        action_indices = [69, 72]
        continuous_obs_indices = [6] + list(range(13, 22)) + list(range(23, 29)) + [30, 31] + list(range(34, 44)) + list(
            range(45, 49)) + list(range(52, 63)) + [71] + list(range(73, 78))
        binary_obs_indices = [5, 49, 50, 70]
        obs_indices = continuous_obs_indices + binary_obs_indices
        obs_action_indices = obs_indices + action_indices

        columns = df_raw.columns.to_numpy()
        columns_obs_action = df_raw.columns[obs_action_indices]
        colums_obs = df_raw.columns[obs_indices]

        #Scale continuous observe data and combine with action data
        df_to_scale = df_raw.iloc[:, continuous_obs_indices]
        df_not_to_scale = df_raw.drop(columns=df_raw.columns[continuous_obs_indices])
        scaled_data = self.scaler.fit_transform(df_to_scale)

        combined_data = np.empty(df_raw.shape, dtype=df_raw.dtypes[0])
        combined_data[:, continuous_obs_indices] = scaled_data
        combined_data[:, np.setdiff1d(np.arange(df_raw.shape[1]), continuous_obs_indices)] = df_not_to_scale.values

        df_combined = pd.DataFrame(combined_data, columns=columns)
        
        grouped = df_combined.groupby('icustayid')
        
        self.data_x = []
        self.data_y = []

        for _, group in grouped:
            if len(group) >= self.max_len:
                group_x = group[columns_obs_action]
                x = group_x.iloc[-self.max_len:-1].to_numpy()
                group_y = group[colums_obs]
                y = group_y.iloc[-1].to_numpy() 
                self.data_x.append(x)
                self.data_y.append(y)

        self.tot_len = len(self.data_x)


    def __getitem__(self, index):
        #seq_x, mask_x = self.pad_sequence(self.data_x[index])
        seq_x = self.data_x[index]
        mask_x = np.zeros(self.max_len)
        seq_y = self.data_y[index]
        
        return seq_x, seq_y, mask_x

    def __len__(self):
        return self.tot_len

    def pad_sequence(self, seq):
        padded_sequence = np.zeros((self.max_len, seq.shape[1]))
        padded_sequence[-seq.shape[0]:, :] = seq  # Pad at the beginning

        mask = np.ones(self.max_len)
        mask[-seq.shape[0]:] = 0

        return padded_sequence, mask
