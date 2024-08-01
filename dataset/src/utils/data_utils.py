
import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, List

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    return tensor
import numpy as np
from typing import Dict, List, Tuple

def create_sequences(data: pd.DataFrame, config: Dict) -> List[Tuple[np.ndarray, float, np.ndarray, int]]:
    window_size = config['data'].get('window_size', 1)
    stride = config['data'].get('stride', 1)
    feature_columns = config['data'].get('feature_columns', 
        config['data']['numerical_columns'] + config['data']['binary_columns'])
    target_column = config['data']['target_column']
    pad_value = config['data'].get('pad_value', 0)
    aggregation = config['data'].get('aggregation')

    # Convert DataFrame to NumPy array for faster operations
    data_array = data[feature_columns + [target_column]].values
    indices = data.index.values

    sequences = []
    for end in range(len(data), 0, -stride):
        start = max(0, end - window_size)
        window = data_array[start:end]
        
        if aggregation and window_size > 1:
            window = aggregate_window(window, config, feature_columns, target_column)

        features = window[:, :-1]  # All columns except the last (target)
        target = window[-1, -1]    # Last row, last column (target)
        mask = [1] * len(window)
        orig_idx = indices[end - 1]

        if len(window) < window_size and not aggregation:
            pad_size = window_size - len(window)
            padding = np.full((pad_size, len(feature_columns)), pad_value)
            features = np.vstack((padding, features))
            mask = [0] * pad_size + mask

        sequences.append((features, target, mask, orig_idx))

    return sequences[::-1]

def aggregate_window(window: np.ndarray, config: Dict, feature_columns: List[str], target_column: str) -> np.ndarray:
    agg_methods = config['data']['aggregation'].get('methods', {})
    
    aggregated = []
    for i, col in enumerate(feature_columns + [target_column]):
        if col in config['data']['numerical_columns']:
            method = agg_methods.get('numerical_columns', 'mean')
        elif col in config['data']['binary_columns']:
            method = agg_methods.get('binary_columns', 'max')
        elif col == target_column:
            method = agg_methods.get('target_column', 'last')
        else:
            method = agg_methods.get('other_columns', 'last')

        if method == 'mean':
            aggregated.append(np.mean(window[:, i]))
        elif method == 'max':
            aggregated.append(np.max(window[:, i]))
        elif method == 'last':
            aggregated.append(window[-1, i])
        elif method == 'first':
            aggregated.append(window[0, i])
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

    return np.array(aggregated).reshape(1, -1)
    
# def create_sequences(data: pd.DataFrame, window_size: int, stride: int, feature_columns: List[str],
#                      target_column: str, pad_value: int = 0) -> List[Tuple[np.ndarray, float, np.ndarray, int]]:
#     """
#     Handles backward sequence creation to accommodate
#     varying lengths of time series. Assume handling one subject at a time.
#     Always includes the last element when stride > 1.
#     """
#     sequences = []
#     features = data[feature_columns].values
#     target = data[target_column].values
#     orig_idx = data.index

#     # Ensure features is 2D
#     if features.ndim == 1:
#         features = features.reshape(1, -1)
    
#     n_rows, n_features = features.shape

#     # Start from the end and work backwards
#     for end in range(n_rows, 0, -stride):
#         start = max(0, end - window_size)
#         sequence = features[start:end]
        
#         if (end - start) < window_size:
#             pad_size = window_size - (end - start)
#             padding = np.full((pad_size, n_features), pad_value)
#             sequence = np.vstack((padding, sequence))
#             mask = [0] * pad_size + [1] * (end - start)
#         else:
#             mask = [1] * window_size
        
#         sequences.append((sequence, target[end-1], mask, orig_idx[end-1]))

#     # Reverse the list so that it's in ascending order
#     return sequences[::-1]

# def aggregate_data(data: pd.DataFrame, config: Dict) -> pd.DataFrame:
#     resample_rate = config['aggregation'].get('resample_rate', 1)
#     if resample_rate == 1:
#         return data
    
#     agg_methods = config['aggregation'].get('methods', {})
#     agg_dict = {}
    
#     for col in data.columns:
#         if col in config['numerical_columns']:
#             agg_dict[col] = agg_methods.get('numerical_columns', 'mean')
#         elif col in config['binary_columns']:
#             agg_dict[col] = agg_methods.get('binary_columns', 'max')
#         elif col == config['target_column']:
#             agg_dict[col] = agg_methods.get('target_column', 'last')
#         else:
#             agg_dict[col] = 'last'
            
#     def create_groups(index):
#         if isinstance(resample_rate, str):
#             resample_rate_int = int(resample_rate.replace('H', ''))
#         else:
#             resample_rate_int = resample_rate
        
#         n = len(index)
#         reversed_groups = (n - 1 - np.arange(n)) // resample_rate_int
#         return pd.Series(reversed_groups.max() - reversed_groups, index=index)
    
#     groups = create_groups(data.index)

#     # Add a helper column to maintain original index
#     data.loc[:, 'original_index'] = data.index
#     agg_dict['original_index'] = 'last'
#     aggregated_data = data.groupby(groups).agg(agg_dict)
#     aggregated_data.index = aggregated_data['original_index']
    
#     # Remove the temporary column
#     aggregated_data = aggregated_data.drop('original_index', axis=1)
    
#     return aggregated_data
