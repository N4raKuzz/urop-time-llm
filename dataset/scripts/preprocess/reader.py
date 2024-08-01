import pandas as pd
from typing import Optional, List
from pandas import DataFrame

class BaseTimeSeriesReader:
    def __init__(self, config: dict):
        self.config = config
        self.file_path = self.config['data_path']
        self.id_col = self.config['columns']['id_col']
        self.index_col = self.config['columns']['index_col']
        self.all_columns = self._get_all_columns()
        self.data = None

    def _get_all_columns(self):
        columns = []
        for col_type in ['numerical', 'binary', 'categorical', 'target']:
            columns.extend(self.config['columns'].get(col_type, []))
        return list(set(columns))

    def _load_raw_data(self) -> None:
        self.data = pd.read_pickle(self.file_path)
        self.data.sort_values(by=[self.id_col, self.index_col], inplace=True)
        self.data.set_index([self.id_col, self.index_col], inplace=True)

    def _filter_data(self) -> None:
        """
        Apply condition-specific filters to the data attribute.
        Assumes data has been loaded with load_data() and is stored in self.data.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def _post_process(self) -> None:
        # Implement any post-processing steps here (e.g., shuffling, reordering)
        pass

    def _extract_relevant_columns(self) -> None:
        self.data = self.data[self.all_columns]

    def get_formatted_data(self) -> DataFrame:
        self._load_raw_data()
        self._filter_data()
        self._post_process()
        self._extract_relevant_columns()
        return self.data

class EFPredReader(BaseTimeSeriesReader):
    def __init__(self, config: dict):
        super().__init__(config)
        self.obs_window = config['reader_params'].get('obs_window', 4)
        self.filter_params = config['reader_params'].get('filter_params', {
            'min_hours_before_extubation': 2,
            'min_LOV': 12,            
            'max_LOV': 672,
            'exclude_deceased': True
        })

    def _filter_data(self) -> None:
        if self.filter_params.get('exclude_deceased'):
            self.data = self.data[self.data['ifDeceased'] != 1]
        
        if 'min_hours_before_extubation' in self.filter_params:
            min_hours = self.filter_params['min_hours_before_extubation']
            self.data = self.data[self.data['first_extubation_time'] - self.data['vent_start_time'] > min_hours]
        
        if 'min_LOV' in self.filter_params:
            min_hours = self.filter_params['min_LOV']
            self.data = self.data[self.data['LOV'] >= min_hours]

        if 'max_LOV' in self.filter_params:
            max_hours = self.filter_params['max_LOV']
            self.data = self.data[self.data['LOV'] <= max_hours]

        # Filter rows: timestamps should be within vent_start_time and first_extubation_time
        self.data = self.data[
            (self.data.index.get_level_values(1) >= self.data['vent_start_time']) &
            (self.data.index.get_level_values(1) < self.data['first_extubation_time'])
        ]

        # Filter to include only the relevant observation window before the first extubation
        def filter_obs_window(group):
            cutoff_start_time = max(group['first_extubation_time'].iloc[0] - self.obs_window, 0)
            cutoff_end_time = group['first_extubation_time'].iloc[0]
            return group[(group.index.get_level_values(1) >= cutoff_start_time) & 
                         (group.index.get_level_values(1) < cutoff_end_time)]

        self.data = self.data.groupby(level=0).apply(filter_obs_window).reset_index(level=0, drop=True)
    
    def _post_process(self):
         self.data['first_extubated_failed'] = (self.data['num_intub'] >= 2).astype(int)

    def get_formatted_data(self):
        return super().get_formatted_data()

class DynamicPredReader(BaseTimeSeriesReader):
    def __init__(self, config: dict):
        super().__init__(config)
        self.filter_params = config['reader_params'].get('filter_params', {})

    def _filter_data(self) -> None:
        if self.filter_params.get('exclude_deceased'):
            self.data = self.data[self.data['ifDeceased'] != 1]
        
        if 'min_hours_before_extubation' in self.filter_params:
            min_hours = self.filter_params['min_hours_before_extubation']
            self.data = self.data[self.data['first_extubation_time'] - self.data['vent_start_time'] > min_hours]
        
        if 'min_LOV' in self.filter_params:
            min_hours = self.filter_params['min_LOV']
            self.data = self.data[self.data['LOV'] >= min_hours]

        if 'max_LOV' in self.filter_params:
            max_hours = self.filter_params['max_LOV']
            self.data = self.data[self.data['LOV'] <= max_hours]

        # Filter rows: timestamps should be within vent_start_time and first_extubation_time
        self.data = self.data[(self.data.index.get_level_values(1) >= self.data['vent_start_time'])]

        if 'min_hours_before_extubation' in self.filter_params: # suggesting we should take extubation time as end point
            self.data = self.data[(self.data.index.get_level_values(1) < self.data['first_extubation_time'])]
            
    def _post_process(self):
        delta_t = 12

        # Define the condition for successful extubation in the next 12 hours
        self.data['successful_extubation_next_12h'] = (
            (self.data.index.get_level_values(1) >= (self.data['vent_end_time'] - delta_t)) &
            (self.data['event_description'].isin(['No 48h Observation - Alive', 'Successful Extubation']) &
            (self.data.num_intub == 1))  # a few cases has failed the first extubation but still extubated within next 12h, we treat these as data error
        ).astype(int)

    def get_formatted_data(self) -> DataFrame:
        return super().get_formatted_data()