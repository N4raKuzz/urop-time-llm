import os
import numpy as np
import pandas as pd

class DataProcessor:
    def __init__(self, data_dir, output_dir, heq_bins=1000, if_continuous_action=True):
        self.heq_bins = heq_bins
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.data = None
        self.outcome = None
        self.action_discrete = None
        self.if_continuous_action = if_continuous_action
        self.action_indices = [69, 72]
        self.continuous_obs_indices = [1,6] + list(range(13, 22)) + list(range(23, 29)) + [30, 31] + list(range(34, 44)) + list(
            range(45, 49)) + list(range(52, 63)) + [71] + list(range(73, 78))
        self.binary_obs_indices = [5, 49, 50, 70]
        self.obs_indices = self.continuous_obs_indices + self.binary_obs_indices
        self.obs_action_indices = self.obs_indices + self.action_indices
        self.terminal_indices = [9, 11]
        

    # Function to save the dataset
    def save_dataset(self, data, output_path, columns):
        df_tosave = pd.DataFrame(data, columns=columns)
        df_tosave.to_csv(output_path, index=False)

    def load_data(self):
        data_path = os.path.join(data_dir, 'MIMICtable_261219.csv')
        outcome_path = os.path.join(data_dir, 'MIMIC_90.csv')
        self.data = pd.read_csv(data_path)
        self.outcome = pd.read_csv(outcome_path)

    def process_mor(self, mor_len=24, mor_gap=6):
        columns = self.data.columns.to_numpy()
        columns_obs_action = columns[self.obs_action_indices]

        grouped = self.data.groupby('icustayid')
        data_to_df = []
        print(len(grouped))
        for _, group in grouped:
            if len(group) > mor_len:
                group_data = group[columns_obs_action]
                total_rows = len(group_data)
                
                # Calculate indices
                indices = np.arange(total_rows-1, 0, -mor_gap)[:mor_len]
                indices = indices[::-1] 
                indices = indices[indices >= 0]
                
                data = group_data.iloc[indices].to_numpy()
        
                icustayid = group['icustayid'].iloc[0] 
                mor_90 = self.outcome[self.outcome['icustay_id'] == icustayid]['morta_90'].values[0]
                mor_90_column = np.zeros((len(data), 1))
                mor_90_column[-4:] = mor_90
 
                data_with_mor90 = np.hstack((data, mor_90_column))
                data_to_df.append(data_with_mor90)

        data_to_df = np.vstack(data_to_df)
        columns_obs_action = np.append(columns_obs_action, 'morta_90')
        self.save_dataset(data_to_df, 'MIMIC_MOR_'+str(mor_len)+'_'+str(mor_gap)+'.csv', columns_obs_action)
        
    def process_seq(self, seq_len=8):
        columns = self.data.columns.to_numpy()
        columns_obs_action = columns[self.obs_action_indices]

        grouped = self.data.groupby('icustayid')
        data_to_df = []

        for _, group in grouped:
            if len(group) > seq_len:
                group_data = group[columns_obs_action]
                start_index = max(0, (len(group) - seq_len) // 2)
                end_index = start_index + seq_len
                data = group_data.iloc[start_index:end_index].to_numpy()
                data_to_df.append(data)

        self.save_dataset(data_to_df, 'MIMIC_'+str(seq_len)+'_1.csv', columns_obs_action)

    def run(self, flag='default'):
        self.load_data()
        if flag == 'default':
            self.process_seq(seq_len=8)
        elif flag == 'mor':
            self.process_mor(mor_len=48, mor_gap=6)


if __name__ == "__main__":
    data_dir = "./orig_data/"
    output_dir = "./processed/"

    mor_processor = DataProcessor(data_dir, output_dir)
    mor_processor.run('mor')