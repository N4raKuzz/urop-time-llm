from typing import Dict, Any, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .reader import EFPredReader, DynamicPredReader
from .preprocessor import Aggregator, Standardizer, SequenceHandler

class DataHandler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reader = self._get_reader()
        self.aggregator = Aggregator(config)
        self.standardizer = Standardizer(config)
        self.sequence_handler = SequenceHandler(config)
        self.target_cols = config['columns'].get('target', [])
        self.all_columns = self.reader.all_columns

    def _get_reader(self):
        readers = {
            'ef_prediction': EFPredReader,
            'dynamic_prediction': DynamicPredReader
        }
        reader_class = readers.get(self.config['task'])
        if not reader_class:
            raise ValueError(f"Unsupported task: {self.config['task']}")
        return reader_class(self.config)

    def _get_indices(self, data):
        return data.index.to_frame(index=False)


    def _separate_features_targets(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Separates features and targets, handling both 2D and 3D inputs.
        For 3D inputs, it extracts the target from the last time step.
        this can't handle non-padding situation so far
        '''
        if not (isinstance(data, pd.DataFrame) or isinstance(data, np.ndarray)):
            raise ValueError("Unsupported data type for feature-target separation")
        
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        target_idx = self.all_columns.index(self.target_cols[0])  # assuming single column target so far
        features = np.delete(data, target_idx, axis=-1)
        targets = data[..., target_idx]

        # Check if we're dealing with 3D data (LSTM-like input)
        if data.ndim == 3:
            targets = targets[:, -1]

        return features, targets
        
    def prepare_data(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        raw_data = self.reader.get_formatted_data()
        agg_data = self.aggregator.aggregate(raw_data)

        # Split data based on ID
        assert agg_data.index.nlevels == 2, "DataFrame must have a two-level index for this operation."
        unique_ids = agg_data.index.get_level_values(0).unique()
        train_val_ids, test_ids = train_test_split(
            unique_ids,
            test_size=0.2
        )
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=0.2
        )

        train = agg_data.loc[train_ids]
        val = agg_data.loc[val_ids]
        test = agg_data.loc[test_ids]

        # Preprocess data
        self.standardizer.fit(train)
        train_std = self.standardizer.transform(train)
        val_std = self.standardizer.transform(val)
        test_std = self.standardizer.transform(test)

        # Update all_columns with new column names
        self.all_columns = train_std.columns.tolist()
        train_indices = self._get_indices(train_std)
        val_indices = self._get_indices(val_std)
        test_indices = self._get_indices(test_std)

        if self.config['sequence']['context_length'] != 1:
            train_std, train_mask = self.sequence_handler.create_sequences(train_std)
            val_std, val_mask = self.sequence_handler.create_sequences(val_std)
            test_std, test_mask = self.sequence_handler.create_sequences(test_std)
        else:
            train_mask, val_mask, test_mask = None, None, None

        train_X, train_y = self._separate_features_targets(train_std)
        val_X, val_y = self._separate_features_targets(val_std)
        test_X, test_y = self._separate_features_targets(test_std)

        return (
            {'X': train_X, 'y': train_y, 'mask': train_mask, 'indices': train_indices},
            {'X': val_X, 'y': val_y, 'mask': val_mask, 'indices': val_indices},
            {'X': test_X, 'y': test_y, 'mask': test_mask, 'indices': test_indices}
        )

    def get_preprocessor(self) -> Dict[str, Any]:
        return {
            'aggregator': self.aggregator,
            'standardizer': self.standardizer,
            'sequence_handler': self.sequence_handler
        }