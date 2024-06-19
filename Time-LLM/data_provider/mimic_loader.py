import torch
import torch.nn as nn
from math import sqrt

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_MIMIC(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='MIMICtable_261219.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.tot_len = 0
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        action_indices = [69, 72]
        continuous_obs_indices = [6] + list(range(13, 22)) + list(range(23, 29)) + [30, 31] + list(range(34, 44)) + list(
            range(45, 49)) + list(range(52, 63)) + [71] + list(range(73, 78))
        binary_obs_indices = [5, 49, 50, 70]
        obs_indices = continuous_obs_indices + binary_obs_indices
        obs_action_indices = obs_indices + action_indices

        columns = df_raw.columns[obs_action_indices]
        grouped = df_raw.groupby('icustayid')
        
        self.data_x = []
        self.data_y = []

        for _, group in grouped:
            self.tot_len += 1
            group = group[columns]
            x = group.iloc[:-1].to_numpy()  # Convert DataFrame to NumPy array
            y = group.iloc[-1].to_numpy()  # Convert Series to NumPy array
            self.data_x.append(x)
            self.data_y.append(y)

        print(f'total steps: {self.tot_len}')


    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        return seq_x, seq_y

    def __len__(self):
        return self.tot_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
