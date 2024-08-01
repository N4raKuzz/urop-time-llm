import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import yaml
import os
import json
import joblib

class VentilationPreprocessor:
    def __init__(self, config_path: str):
        """
        Initialize the VentilationPreprocessor with a configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.data: pd.DataFrame = None
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False)
        self.seed = self.config['split']['random_state']
        self.exclusion_records = {}

    def load_data(self) -> None:
        """Load the data from the specified input path."""
        self.data = pd.read_pickle(self.config['data']['input_path'])
        self.data.sort_values(by=[self.config['data']['id_column'], self.config['data']['index_column']], inplace=True)
        self.data.set_index([self.config['data']['id_column'], self.config['data']['index_column']], inplace=True)

    def apply_filters(self) -> None:
        """Apply filtering rules specified in the configuration."""
        for rule in self.config['preprocessing']['excluding']['rules']:
            column = rule['column']
            operation = rule['operation']
            value = rule['value']
            n_subjects = self.data.index.get_level_values(0).nunique()

            if operation == '==':
                self.data = self.data[self.data[column] != value]
            elif operation == '<=':
                self.data = self.data[self.data[column] > value]
            elif operation == '>':
                self.data = self.data[self.data[column] <= value]
            # Add more operations as needed

            print(f"Subjects excluded by rule ({column} {operation} {value}): {n_subjects - self.data.index.get_level_values(0).nunique()}")
            self.exclusion_records[f"{column} {operation} {value}"] = n_subjects - self.data.index.get_level_values(0).nunique()

    def preprocess(self) -> None:
        """Preprocess the data: standardize numerical columns, encode categorical columns, and center binary columns."""
        num_cols = self.config['data']['columns']['numerical']
        cat_cols = self.config['data']['columns']['categorical']
        bin_cols = self.config['data']['columns']['binary']

        # Standardize numerical columns
        self.data[num_cols] = self.scaler.fit_transform(self.data[num_cols])

        # Encode categorical columns
        encoded_cats = self.encoder.fit_transform(self.data[cat_cols])
        encoded_cat_cols = self.encoder.get_feature_names_out(cat_cols)
        self.data = pd.concat([self.data, 
                               pd.DataFrame(encoded_cats, columns=encoded_cat_cols, index=self.data.index)], axis=1)

        # Update binary columns to include one-hot encoded categorical columns
        bin_cols.extend(encoded_cat_cols)

        # Center binary columns
        self.data[bin_cols] = self.data[bin_cols] * 2 - 1

    def feature_engineering(self) -> None:
        pass

    def split_data(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Split the data into train, validation, and test sets.

        Returns:
            Tuple[List[str], List[str], List[str]]: Lists of IDs for train, validation, and test sets.
        """
        unique_ids = self.data.index.get_level_values(0).unique()
        train_val_ids, test_ids = train_test_split(unique_ids, test_size=self.config['split']['test_size'], random_state=self.seed)
        train_ids, val_ids = train_test_split(train_val_ids, test_size=self.config['split']['val_size'], random_state=self.seed)

        return list(train_ids), list(val_ids), list(test_ids)

    def save_data(self, train_ids: List[str], val_ids: List[str], test_ids: List[str]) -> None:
        """
        Save the preprocessed data, split information, and metadata.

        Args:
            train_ids (List[str]): List of IDs for the training set.
            val_ids (List[str]): List of IDs for the validation set.
            test_ids (List[str]): List of IDs for the test set.
        """
        output_dir = f"{self.config['data']['output_dir']}_seed{self.seed}"
        os.makedirs(output_dir, exist_ok=False)

        # Save data for each split
        for split, ids in zip(['train', 'val', 'test'], [train_ids, val_ids, test_ids]):
            split_dir = os.path.join(output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            for subject_id in ids:
                subject_data = self.data.loc[subject_id]
                file_name = f"{subject_id}.parquet"
                subject_data.to_parquet(os.path.join(split_dir, file_name))

        # Save split information
        splits = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
        with open(os.path.join(output_dir, 'splits.json'), 'w') as f:
            json.dump(splits, f)

        # Save metadata
        metadata = {
            'seed': self.seed,
            'columns': list(self.data.columns),
            'id_column': self.config['data']['id_column'],
            'index_column': self.config['data']['index_column'],
            'exclusion_records': self.exclusion_records
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        # Save scalers and encoders
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.joblib'))
        joblib.dump(self.encoder, os.path.join(output_dir, 'encoder.joblib'))

    def run(self) -> None:
        """Execute the entire preprocessing pipeline."""
        self.load_data()
        self.apply_filters()
        self.preprocess()
        train_ids, val_ids, test_ids = self.split_data()
        self.save_data(train_ids, val_ids, test_ids)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to data config file', \
        default='scripts/preprocess/preprocessing_settings_ventilation.yaml')
    args = parser.parse_args()
    
    preprocessor = VentilationPreprocessor(args.config)
    preprocessor.run()

    print('Preprocessing completed. Data saved in the specified output directory.')