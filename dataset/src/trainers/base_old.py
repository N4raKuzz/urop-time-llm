from abc import ABC, abstractmethod
from typing import Any, Dict, List
from .experiment_config import ExperimentConfig
from ..utils.experiment_utils import (
    set_global_seed
)
from ..utils.result_tracker import ResultTracker

class BaseExperiment(ABC):
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.result_tracker = ResultTracker(results_dir=config.results_dir)
        self._data_dict = None

    @property
    def data_dict(self):
        if self._data_dict is None:
            self._data_dict = self.prepare_data()
        return self._data_dict

    @abstractmethod
    def prepare_data(self) -> Any:
        pass

    def prepare_data_loader(self, data: Any) -> Any:
        pass

    @abstractmethod
    def setup_model(self) -> Any:
        pass

    @abstractmethod
    def train(self, model: Any, data: Any) -> Any:
        pass

    @abstractmethod
    def evaluate(self, model: Any, data: Any) -> Dict[str, Any]:
        pass

    def reset_data_dict(self):
        self._data_dict = None   
    
    def run(self, seed: int) -> Dict[str, Any]:
        set_global_seed(seed)
        self.config.seed = seed
        try:
            data_loaders = self.prepare_data_loader(self.data_dict)

            # tmp for TTE analysis
            # Save data_dict to a pickle file
            # import os, pickle
            # output_dir = './data/tmp_tte'
            # os.makedirs(output_dir, exist_ok=True)
            # output_path = os.path.join(output_dir, f'prepared_ttedata_{seed}_{self.config.task}.pkl')
            
            # with open(output_path, 'wb') as f:
            #     pickle.dump(self.data_dict, f)
            #####

            self.config.model_config['input_dim'] = self.data_dict[0]['X'].shape[-1]
            model = self.setup_model()
            trained_model, trained_logger = self.train(model, data_loaders)
            results = self.evaluate(trained_model, data_loaders)
            return results, trained_logger
        except Exception as e:
            print(f"An error occurred during the experiment: {str(e)}")
            return {"error": str(e)}

    def run_multiple_seeds(self, seeds: List[int]) -> List[Dict[str, Any]]:
        all_results = []
        for seed in seeds:
            results, logger = self.run(seed)
            all_results.append(results)
            self.save_results(results, seed)
            self.print_results(results, seed)

        return all_results

    def save_results(self, results: Dict[str, Any], seed: int) -> None:
        self.result_tracker.save_result(results, self.config, seed)

    def print_results(self, results: Dict[str, Any], seed: int) -> None:
        print(f"Results for seed {seed}:")
        for model_name, metrics in results['metrics'].items():
            print(f"Model: {model_name}")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")