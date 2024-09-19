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
    def __init__(self, root_path, flag='train', max_len=12,
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
        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path
        self.tot_len = 0
        self.__read_data__()

    def __read_data__(self):
        

        print(f'Reading Dataset {self.data_path} for {self.flag}')

        df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path))
        df_raw = df_raw.sample(frac=1, random_state=42)

        self.columns = df_raw.columns.to_numpy()
        self.columns_scale = df_raw.columns[list(range(2,51))] 
        self.columns_output = df_raw.columns[49]      
        self.columns_input = df_raw.columns[list(range(1,49)) + list(range(50,56))]
        #Scale continuous observe data and combine with action data

        self.scaler = StandardScaler()
        df_to_scale = df_raw[self.columns_scale]
        df_not_to_scale = df_raw.drop(columns=self.columns_scale)
        scaled_data = self.scaler.fit_transform(df_to_scale)

        combined_data = np.empty(df_raw.shape, dtype=df_raw.dtypes[0])
        combined_data[:, list(range(2,51))] = scaled_data
        combined_data[:, np.setdiff1d(np.arange(df_raw.shape[1]), list(range(2,51)))] = df_not_to_scale.values

        df_combined = pd.DataFrame(combined_data, columns=self.columns)
        
        grouped = df_combined.groupby('icustayid')
        
        train_data_x = []
        train_data_y = []
        test_data_x = []
        test_data_y = []
        vali_data_x = []
        vali_data_y = []

        for _, group in grouped:
            if len(group) > self.max_len:
                group_x = group[self.columns_input]
                x = group_x.iloc[-self.max_len-1:-1].to_numpy()
                # x = x.flatten()
                
                group_y = group[self.columns_output]
                y = (group_y.iloc[-1] > group_y.iloc[-2]).astype(float)

                if len(train_data_x) < 16000:
                    train_data_x.append(x)
                    train_data_y.append(y)
                elif len(test_data_x) < 2000:
                    test_data_x.append(x)
                    test_data_y.append(y)
                elif len(vali_data_x) < 2000:
                    vali_data_x.append(x)
                    vali_data_y.append(y)
                else:
                    break

        if self.flag == 'train':
            self.data_x = train_data_x
            self.data_y = train_data_y
        elif self.flag == 'test':
            self.data_x = test_data_x
            self.data_y = test_data_y
        else:
            self.data_x = vali_data_x
            self.data_y = vali_data_y

        self.tot_len = len(self.data_x)
        print(f'{self.flag} dataset len: {self.tot_len}')


    def __getitem__(self, index):
        # seq_x, mask_x = self.pad_sequence(self.data_x[index])
        seq_x = self.data_x[index]
        # mask_x = np.zeros(self.max_len)
        seq_y = self.data_y[index]
        
        return seq_x, seq_y

    def __len__(self):
        return self.tot_len
    
    def get_scaler(self):
        return self.scaler
    
    def get_columns(self):
        return self.columns_input, self.columns_output

    def pad_sequence(self, seq):
        padded_sequence = np.zeros((self.max_len, seq.shape[1]))
        padded_sequence[-seq.shape[0]:, :] = seq  # Pad at the beginning

        mask = np.ones(self.max_len)
        mask[-seq.shape[0]:] = 0

        return padded_sequence, mask

class Dataset_MOR(Dataset):
    def __init__(self, root_path, flag='train', max_len=12,
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
        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path
        self.tot_len = 0
        self.target_ratio = 2
        self.__read_data__()

    def __read_data__(self):
    
        print(f'Reading Dataset {self.data_path} for {self.flag}')

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        self.columns = df_raw.columns.to_numpy()
        self.columns_scale = df_raw.columns[list(range(2,51))] 
        self.columns_output = df_raw.columns[56]      
        self.columns_input = df_raw.columns[list(range(1,56))]
        #Scale continuous observe data and combine with action data

        self.scaler = StandardScaler()
        df_to_scale = df_raw[self.columns_scale]
        df_not_to_scale = df_raw.drop(columns=self.columns_scale)
        scaled_data = self.scaler.fit_transform(df_to_scale)

        combined_data = np.empty(df_raw.shape, dtype=df_raw.dtypes[0])
        combined_data[:, list(range(2,51))] = scaled_data
        combined_data[:, np.setdiff1d(np.arange(df_raw.shape[1]), list(range(2,51)))] = df_not_to_scale.values

        df_combined = pd.DataFrame(combined_data, columns=self.columns)
        
        df_y0 = df_combined[df_combined[self.columns_output] == 0]
        df_y1 = df_combined[df_combined[self.columns_output] == 1]
        n_y1 = len(df_y1)
        n_y0_keep = min(len(df_y0), n_y1 * self.target_ratio)

        df_y0_downsampled = df_y0.sample(n=int(n_y0_keep), random_state=42)
        df_balanced = pd.concat([df_y0_downsampled, df_y1]) 
        df_balanced = df_balanced.sample(frac=1, random_state=42)        

        train_data_x = []
        train_data_y = []
        test_data_x = []
        test_data_y = []
        vali_data_x = []
        vali_data_y = []

        for _, row in df_balanced.iterrows():
            x = row[self.columns_input].values
            y = row[self.columns_output].astype(float)

            if len(train_data_x) < 35000:
                train_data_x.append(x)
                train_data_y.append(y)
            elif len(test_data_x) < 4500:
                test_data_x.append(x)
                test_data_y.append(y)
            elif len(vali_data_x) < 4500:
                vali_data_x.append(x)
                vali_data_y.append(y)
            else:
                break

        if self.flag == 'train':
            self.data_x = train_data_x
            self.data_y = train_data_y
        elif self.flag == 'test':
            self.data_x = test_data_x
            self.data_y = test_data_y
        else:
            self.data_x = vali_data_x
            self.data_y = vali_data_y

        self.tot_len = len(self.data_x)
        print(f'{self.flag} dataset len: {self.tot_len}')


    def __getitem__(self, index):
        # seq_x, mask_x = self.pad_sequence(self.data_x[index])
        seq_x = self.data_x[index]
        # mask_x = np.zeros(self.max_len)
        seq_y = self.data_y[index]
        
        return seq_x, seq_y

    def __len__(self):
        return self.tot_len
    
    def get_scaler(self):
        return self.scaler
    
    def get_columns(self):
        return self.columns_input, self.columns_output


