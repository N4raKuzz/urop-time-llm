import os
import numpy as np
import collections
import pickle
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

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
        self.continuous_obs_indices = [6] + list(range(13, 22)) + list(range(23, 29)) + [30, 31] + list(range(34, 44)) + list(
            range(45, 49)) + list(range(52, 63)) + [71] + list(range(73, 78))
        self.binary_obs_indices = [5, 49, 50, 70]
        self.obs_indices = self.continuous_obs_indices + self.binary_obs_indices
        self.obs_action_indices = self.obs_indices + self.action_indices
        self.terminal_indices = [9, 11]

        # Caps for actions - vaso and ivf
        self.caps = {'vaso': 1.5, 'ivf': 1250.}

    # Function to save the dataset
    def save_dataset(self, data_tosave, output_file):
        with open(output_file, 'wb') as f:
            pickle.dump(data_tosave, f)

    def load_data(self):
    # load data and outcome from respective paths
        data_path = os.path.join(data_dir, 'MIMICtable_261219.csv')
        outcome_path = os.path.join(data_dir, 'MIMIC_90.csv')
        self.data = np.genfromtxt(data_path, delimiter=',')
        self.outcome = np.genfromtxt(outcome_path, delimiter=',')
        # Remove column names
        self.data = self.data[1:]
        self.outcome = self.outcome[1:]

    def preprocess_data(self):
    # preprocess data, including quantile transformation
    # save scalers as necessary

        # Prepare continuous observations
        scaler_cont_obs = QuantileTransformer()
        scaler_cont_obs.fit(self.data[:, self.continuous_obs_indices])
        self.data[:, self.continuous_obs_indices] = scaler_cont_obs.transform(self.data[:, self.continuous_obs_indices])
        self.save_dataset(scaler_cont_obs, os.path.join(self.output_dir, f'scalers/qt_contobs.obj'))

        # Prepare actions
        num_actions = len(self.action_indices)
        self.action_discrete = np.zeros((len(self.data), num_actions), dtype=int)

        for i, (action, cap) in enumerate(self.caps.items()):
            non_zero_indices = np.where(self.data[:, self.action_indices[i]] > 0)[0]

            scaler_action = QuantileTransformer(n_quantiles=self.heq_bins)
            scaler_action.fit(np.clip(self.data[non_zero_indices, self.action_indices[i]].reshape(-1, 1), 0., cap))

            self.action_discrete[non_zero_indices, i] = np.digitize(
                np.clip(self.data[non_zero_indices, self.action_indices[i]], 0., cap), scaler_action.quantiles_.flatten())

            self.data[non_zero_indices, self.action_indices[i]] = scaler_action.transform(
                self.data[non_zero_indices, self.action_indices[i]].reshape(-1, 1)).flatten() + 3 / self.heq_bins

            filename = os.path.join(output_dir,f'scalers/qt_contobs_{action}.obj')
            self.save_dataset(scaler_action, filename)

    def parse_to_episodes(self):
    # divide data into episodes
    # return episodes, action_index_epi, terminals

        # divide into episodes
        episodes = []
        actions = []
        terminals = []  # 0: discharge  1: death
        startind = 0
        ipp = 0

        for i_pt in range(1, len(self.data)):
            if self.data[i_pt, 0] <= self.data[i_pt - 1, 0]:
                if self.data[startind, 1] != self.outcome[ipp, 2]:
                    print("id not matched !")
                else:
                    self.data[startind:i_pt, self.terminal_indices[1]] = self.outcome[ipp, -1]
                ipp += 1
                # Only add episodes with length >= 2
                if i_pt - startind >= 2:
                    episodes.append(np.array(self.data[startind:i_pt, self.obs_action_indices]))
                    actions.append(self.action_discrete[startind:i_pt].copy())
                    if np.sum(self.data[i_pt - 1, self.terminal_indices]) == 0:
                        terminals.append(0)
                    else:
                        terminals.append(1)
                startind = i_pt
        if self.data[startind, 1] != self.outcome[ipp, 2]:
            print("id not matched !")
        else:
            self.data[startind:, self.terminal_indices[1]] = self.outcome[ipp, -1]

        # Only add episodes with length >= 2
        if len(self.data) - startind >= 2:
            episodes.append(np.array(self.data[startind:, self.obs_action_indices]))
            actions.append(self.action_discrete[startind:].copy())
            if np.sum(self.data[-1, self.terminal_indices]) == 0:
                terminals.append(0)
            else:
                terminals.append(1)

        print(len(terminals), "patient trajectories collected")
        return episodes, actions, np.array(terminals)

    def parse_to_transitions(self, episodes, actions, terminals, file_name):
    # parse episodes into transitions and save to file_name
    # episodes include continuous actions, while "actions" is discrete, for DT method, we use continuous one

        rewards = -2. * np.array(terminals) + 1.
        data_ = collections.defaultdict(list)
        num_actions = len(self.action_indices)
        paths = []

        for i in range(len(terminals)):
            traj_length = episodes[i].shape[0]

            data_['observations'] = np.array(episodes[i][:, :-num_actions])
            data_['next_observations'] = np.array(
                np.concatenate((episodes[i][1:, :-num_actions], np.zeros_like(episodes[i][-1:, :-num_actions])),
                               axis=0))  # zero padding
            if self.if_continuous_action:
                data_['actions'] = np.array(
                    np.concatenate((episodes[i][:-1, -num_actions:], np.zeros_like(episodes[i][-1:, -num_actions:])),
                                   axis=0))  # zero padding
            else:
                data_['actions'] = np.array(
                    np.concatenate((actions[i][:-1, :], np.zeros_like(actions[i][-1:, :])), axis=0))# zero padding

            data_['rewards'] = np.array([0] * (traj_length - 1) + [rewards[i]])
            data_['terminals'] = np.array([False] * (traj_length - 1) + [True])

            paths.append(data_)
            data_ = collections.defaultdict(list)

        self.save_dataset(paths, os.path.join(self.output_dir, file_name))
        num_samples = np.sum([p['rewards'].shape[0] for p in paths])
        print(f'{num_samples} transitions created from {len(paths)} trajectories')

    def split_train_test(self, episodes, actions, terminals, test_ratio, seed, if_save_episode):
    # split episodes, action_index_epi, and terminals into train and test sets
    # according to the given test_ratio

        indices = np.arange(len(episodes))

        train_episodes, test_episodes, train_actions, test_actions, train_terminals, test_terminals, train_indices,\
            test_indices= train_test_split(episodes, actions, terminals, indices, test_size=test_ratio, random_state=seed)

        if if_save_episode:
            self.save_dataset([train_episodes, train_actions, train_terminals], \
                         os.path.join(self.output_dir, f'Data_ActInd_Term_epi.pkl'))
            self.save_dataset([test_episodes, test_actions, test_terminals], \
                         os.path.join(self.output_dir, f'Data_ActInd_Term_epi_test.pkl'))

        self.save_dataset(test_indices, os.path.join(self.output_dir, f'Indices_test.pkl'))

        suffix = "" if self.if_continuous_action else "_Discrete"
        self.parse_to_transitions(train_episodes, train_actions, train_terminals,
                                  f'Transition_Data{suffix}.pkl')
        self.parse_to_transitions(test_episodes, test_actions, test_terminals,
                                  f'Transition_Data{suffix}_test.pkl')

    def run(self, test_ratio=0.1, seed=66, if_save_episode=False):
        self.load_data()
        self.preprocess_data()
        episodes, actions, terminals = self.parse_to_episodes()
        self.split_train_test(episodes, actions, terminals, test_ratio, seed, if_save_episode)

if __name__ == "__main__":
    data_dir = "./orig_data/"
    output_dir = "./processed/"
    test_ratio = 0.1
    seed = 66

    aic_processor = DataProcessor(data_dir, output_dir, heq_bins=5, if_continuous_action=True)
    aic_processor.run(test_ratio, seed, if_save_episode=True)
    dt_processor = DataProcessor(data_dir, output_dir, heq_bins=5, if_continuous_action=False)
    dt_processor.run(test_ratio, seed, if_save_episode=False)