import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Optional

class Aggregator:
    def __init__(self, config: dict):
        """
        Initialize the Aggregator with a configuration dictionary.
        """
        self.config = config
        self._setup_config()

    def _setup_config(self):
        self.numerical_cols = self.config['columns'].get('numerical', [])
        self.categorical_cols = self.config['columns'].get('categorical', [])
        self.binary_cols = self.config['columns'].get('binary', [])
        self.target_cols = self.config['columns'].get('target', [])
        self.resample_rate = self.config['aggregation']['resample_rate']
        self.numerical_aggfunc = self.config['aggregation']['methods'].get('numerical', 'mean')
        self.categorial_aggfunc = self.config['aggregation']['methods'].get('categorical', 'last')
        self.binary_aggfunc = self.config['aggregation']['methods'].get('binary', 'max')
        self.target_aggfunc = self.config['aggregation']['methods'].get('target', 'last')

    def _create_relative_groups(self, index):
        if isinstance(self.resample_rate, str):
            self.resample_rate = int(self.resample_rate.replace('H', ''))
        
        def group_indices(group):
            n = len(group)
            reversed_groups = (n - 1 - np.arange(n)) // self.resample_rate
            return pd.Series(reversed_groups.max() - reversed_groups, index=group)
        
        return index.to_frame().groupby(level=0).apply(lambda x: group_indices(x.index.get_level_values(1))).values

    def aggregate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data based on the specified configuration for aggregation.
        Assuming data is sorted and aligned.
        """
        assert data.index.nlevels == 2, "DataFrame must have a two-level index for this operation."
        
        # If resample rate is 1, return the input DataFrame as is
        if self.resample_rate == 1:
            return data
            
        # Aggregate features
        agg_dict = {col: self.numerical_aggfunc for col in self.numerical_cols if col in data.columns}
        agg_dict.update({col: self.categorial_aggfunc for col in self.categorical_cols if col in data.columns})
        agg_dict.update({col: self.binary_aggfunc for col in self.binary_cols if col in data.columns})
        agg_dict.update({col: self.target_aggfunc for col in self.target_cols if col in data.columns})
        
        # Perform aggregation
        relative_groups = self._create_relative_groups(data.index)
        aggregated_data = data.groupby([data.index.get_level_values(0), relative_groups]).agg(agg_dict).dropna()
        
        # to ensure same order of columns
        return aggregated_data[data.columns]

class Standardizer(BaseEstimator, TransformerMixin):
    """
    Preprocessor class for handling different types of data columns (numerical, binary, categorical),
    including configurable aggregation, standardization, and normalization based on a YAML configuration.
    """

    def __init__(self, config: dict):
        """
        Initialize the Preprocessor with configuration dictionary.
        """
        self.config = config
        self._setup_config()
        # Initialize preprocessing components
        self.trained = False
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False)
      

    def _setup_config(self):
        self.numerical_cols = self.config['columns'].get('numerical', [])
        self.categorical_cols = self.config['columns'].get('categorical', [])
        self.binary_cols = self.config['columns'].get('binary', [])
        self.target_cols = self.config['columns'].get('target', [])

    def fit(self, data: pd.DataFrame):
        """
        Fit the preprocessing components to the training data based on specified columns.
        """
        self.binary_cols = [col for col in data.columns if col.startswith(tuple(self.binary_cols))]

        # Fit scaler for numerical columns
        if self.numerical_cols is not None:
            self.scaler.fit(data[self.numerical_cols])

        # Fit encoder for categorical columns
        if self.categorical_cols is not None:
            self.encoder.fit(data[self.categorical_cols])
            # Update binary columns after encoding
            encoded_features = self.encoder.get_feature_names_out(self.categorical_cols)
            self.binary_cols.extend(encoded_features)

        self.trained = True

        return self

    def transform(self, data: pd.DataFrame) -> (pd.DataFrame):
        """
        Apply transformations to the data including aggregation, scaling, and encoding.
        """
        if not self.trained:
            raise ValueError("This Preprocessor instance is not fitted yet.")
        
        assert data.index.nlevels == 2, "DataFrame must have a two-level index for this operation."

        scaled_data = data.copy()

        # Scale numerical columns
        if self.numerical_cols:
            scaled_data[self.numerical_cols] = self.scaler.transform(scaled_data[self.numerical_cols])

        # Encode categorical columns
        if self.categorical_cols:
            encoded_cats = self.encoder.transform(scaled_data[self.categorical_cols])
            cat_cols = self.encoder.get_feature_names_out()
            scaled_data.drop(self.categorical_cols, axis=1, inplace=True)
            scaled_data = pd.concat([scaled_data, pd.DataFrame(encoded_cats, columns=cat_cols, index=scaled_data.index)], axis=1)

        # Center binary columns
        if self.binary_cols:
            scaled_data[self.binary_cols] = scaled_data[self.binary_cols] * 2 - 1

        return scaled_data

class SequenceHandler:
    def __init__(self, config: dict):
        self.config = config
        self._setup_config()
        self.target_indices = None
        self.padded = False

    def _setup_config(self):
        self.context_length = self.config['sequence'].get('context_length', 12)
        self.pad_value = self.config['sequence'].get('pad_value', 0)
        self.target_cols = self.config['columns'].get('target', [])

    def create_sequences(self, data: pd.DataFrame, add_padding: Optional[bool] = True):
        '''this can't handle non-padding situation so far'''
        """
        Create sequences from a multi-level indexed DataFrame, where the outer level is assumed
        to be 'id' and the inner level 'time'. Handles backward sequence creation to accommodate
        varying lengths of time series per id.
        """
        assert data.index.nlevels == 2, "DataFrame must have a two-level index for this operation."

        self.padded = add_padding
        self.target_indices = [data.columns.get_loc(col) for col in self.target_cols if col in data.columns]

        sequences = []
        masks = []
        ids = data.index.get_level_values(0).unique()
        for id_ in ids:
            id_data = data.loc[id_].values
            if id_data.size == 0:  # Skip empty data
                continue
            # Ensure id_data is 2D
            if id_data.ndim == 1:
                id_data = id_data.reshape(1, -1)
            
            n_rows, n_features = id_data.shape

            for end in range(1, n_rows + 1):
                start = max(0, end - self.context_length)
                sequence = id_data[start:end]
                if add_padding and (end - start) < self.context_length:
                    pad_size = self.context_length - (end - start)
                    padding = np.full((pad_size, n_features), self.pad_value)
                    sequence = np.vstack((padding, sequence))
                    mask = [0] * pad_size + [1] * (end - start)
                else:
                    mask = [1] * (end - start)

                sequences.append(sequence)
                masks.append(mask)

        return np.array(sequences), np.array(masks)

    # def separate_features_targets(self, sequences):
    #     """
    #     Separate features and targets from sequences using column indices.
    #     """
    #     features = []
    #     targets = []

    #     for seq in sequences:
    #         if seq.ndim == 2:
    #             feature_indices = [i for i in range(seq.shape[1]) if i not in self.target_indices]
    #             features.append(seq[:, feature_indices])
    #             targets.append(seq[-1, self.target_indices])
    #         else:
    #             raise ValueError("All sequences must be 2-dimensional arrays.")

    #     if self.padded:
    #         features = np.array(features)
        
    #     return features, np.array(targets)

class Discretizer:
    pass