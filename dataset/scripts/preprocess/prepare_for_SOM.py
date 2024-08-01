import sys
from pathlib import Path
# Add the project root directory to the Python path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from data_handler import DataHandler
import yaml
import h5py
import numpy as np
import pandas as pd

class ICUDataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.data_handler = DataHandler(config)
        
    def preprocess(self):
        raw_data = self.data_handler.prepare_data()
        train_data, val_data, test_data = raw_data
        
        self.process_and_save_split(train_data, 'train')
        self.process_and_save_split(val_data, 'val')
        self.process_and_save_split(test_data, 'test')

    def process_and_save_split(self, data, split):
        save_dir = Path(self.config['save_dir']) / split
        save_dir.mkdir(parents=True, exist_ok=True)

        X = data['X']
        y = data['y']
        if self.config['columns']['target'][0] == 'diagnosis':
            y = pd.Series(y)
            cat_to_code = {
                'cardiovascular': 0,
                'respiratory': 1,
                'infection': 2,
                'neurological': 3,
                'gastrointestinal': 4,
                'others':5
            }
            y_categorical = y.map(cat_to_code)
            y = y_categorical.values

        mask = data['mask']
        indices = data['indices']

        id_column = indices.columns[0].astype(str)  # Get the first column name as ID

        for subject in indices[id_column].unique():
            subject_mask = indices[id_column] == subject
            subject_X = X[subject_mask]
            subject_y = y[subject_mask]
            subject_indices = indices[subject_mask].iloc[:, -1]
            subject_mask = mask[subject_mask] if mask is not None else None

            self.save_as_h5(save_dir / f'{subject}.h5', subject_X, subject_y, subject_indices, subject_mask)

    def save_as_h5(self, file_path, data, labels, indices, mask=None):
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('x', data=data.astype(np.float32))
            f.create_dataset('labels', data=labels)
            f.create_dataset('indices', data=indices)
            f.create_dataset('nr_samples', data=data.shape[0])
            f.create_dataset('fs', data=self.config['sampling_rate'])
            if mask is not None:
                f.create_dataset('mask', data=mask)

    def save_meta_data(self):
        meta_data = {
            'sampling_rate': self.config['sampling_rate'],
            'window_size': self.config['sequence']['context_length'],
            'dataset': 'icu_ventilation_data',
            'nr_classes': self.config['nr_classes']
        }
        
        with open(Path(self.config['save_dir']) / 'preprocessing_specifics.yml', 'w') as f:
            yaml.dump(meta_data, f)

if __name__ == "__main__":
    config_path ='config/configs_SOM.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    preprocessor = ICUDataPreprocessor(config)
    preprocessor.preprocess()
    preprocessor.save_meta_data()

    print('Done! Move the subjects to the train, val and test subfolders in the way you want to make your split.')