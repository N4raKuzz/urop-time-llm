import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from src.utils.data_utils import create_sequences, aggregate_window

class VentilationDataset(Dataset):
    '''This dataset contains only timestamps between vent_start_time and first_extubation_time'''
    def __init__(self, config: Dict, data_fold: str = 'train'):
        self.data_fold = data_fold.lower()
        assert self.data_fold in ['train', 'val', 'test']
        
        self.config = config
        self.data_dir = os.path.join(config['data']['preprocessed_dir'], data_fold)
        self.metadata = self._load_metadata()
        self.subject_ids = self._get_subject_ids()
        
        self.feature_columns = config['data'].get('feature_columns', \
            config['data']['numerical_columns'] + config['data']['binary_columns']) 
        self.target_column = config['data']['target_column']
        
        self.window_size = config['data'].get('window_size', 1)
        self.stride = config['data'].get('stride', 1)
        
        self.all_data, self.valid_indices = self._preload_and_process_data()

    def _load_metadata(self) -> Dict:
        with open(os.path.join(self.config['data']['preprocessed_dir'], 'metadata.json'), 'r') as f:
            return json.load(f)

    def _get_subject_ids(self) -> List[str]:
        return [f.split('.')[0] for f in os.listdir(self.data_dir) if f.endswith('.parquet')]

    def _preload_and_process_data(self) -> Tuple[Dict[str, List[Tuple[np.ndarray, float, np.ndarray]]], List[Tuple[str, int]]]:
        processed_data = {}
        valid_indices = []

        for subject_id in self.subject_ids:
            subject_data = pd.read_parquet(os.path.join(self.data_dir, f"{subject_id}.parquet"))

            # get rid of unnecessary data in the sequence to facilitate correct positional handling later
            # e.g. to align all sequence by the first extubation time
            subject_data = self.pre_filter_data(subject_data)

            # Compute task-specific targets
            subject_data = self.compute_targets(subject_data)

            # Create sequences
            sequences = create_sequences(subject_data, self.config)
            processed_data[subject_id] = sequences

            # Match valid_mask with sequences based on original index
            valid_mask = self.filter_data(subject_data)
            for idx, (_, _, _, orig_idx) in enumerate(sequences):
                if orig_idx in subject_data.index[valid_mask].values:
                    valid_indices.append((subject_id, idx, orig_idx))
        
        return processed_data, valid_indices

    def pre_filter_data(self,data: pd.DataFrame) -> np.ndarray:
        '''this could be replaced by the wrapper, in case other logics apply before aggregation/sequence extraction'''
        vent_end_mask = ((data.index < data['first_extubation_time'].iloc[0]) | \
            (data['event_description'].str.startswith('Not Extubated')))
        return data[vent_end_mask]

    def filter_data(self, data: pd.DataFrame) -> np.ndarray:
        # Default implementation: all data is valid
        return np.ones(len(data), dtype=bool)

    def compute_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        # Placeholder for task-specific target computation
        return data

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        subject_id, seq_idx, _ = self.valid_indices[idx]
        sequence, target, _, _ = self.all_data[subject_id][seq_idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32).squeeze()

    def get_data_by_index(self, subject_id: str, orig_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        for _, (sequence, target, _, seq_orig_idx) in enumerate(self.all_data[subject_id]):
            if seq_orig_idx == orig_idx:
                return torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32).squeeze()
        raise ValueError(f"No data found for subject {subject_id} at index {orig_idx}")