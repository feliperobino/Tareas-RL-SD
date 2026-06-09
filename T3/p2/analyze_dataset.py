import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import numpy as np

def defaultList():
    return []

class DatasetAnalyzer:
    def __init__(self, comp_dataset, expert_trajectories):
        self.comp_dataset = comp_dataset
        self.expert_trajectories = expert_trajectories

        self.wines_profiles = self.get_unique(comp_dataset)
        self.characteristics = self.get_characteristics(self.wines_profiles)
        self.expert_averages = self.get_expert_averages(self.expert_trajectories)
        self.plot_characteristics(self.characteristics, self.expert_averages)

    def get_unique(self, dataset):
        wines = {}
        for i in range(len(dataset)):
            elem = dataset[i]
            wines[elem['idx_A']] = elem['profile_A']
            wines[elem['idx_B']] = elem['profile_B']
        return wines

    def get_characteristics(self, wines_profiles):
        characteristics = defaultdict(defaultList)
        for k, v in wines_profiles.items():
            for char_name, char_value in v.items():
                characteristics[char_name].append(char_value)

        return characteristics

    def get_expert_averages(self, expert_trajectories):
        avg_values = defaultdict(defaultList)
        for elem in expert_trajectories:
            for k, v in elem['profile'].items():
                avg_values[k].append(v)

        avg_values_aux = {}
        for k, v in avg_values.items():
            avg_values_aux[k] = float(np.mean(np.array(v)))

        return avg_values_aux

    def plot_characteristics(self, characteristics, expert_values):
        characteristics_names = list(characteristics.keys())
        plt.figure(figsize=(15, 15))
        for i in range(len(characteristics)):
            plt.subplot(3, 3, i + 1)
            x = [j for j in range(len(characteristics[characteristics_names[i]]))]
            plt.scatter(x, characteristics[characteristics_names[i]], label=characteristics_names[i])
            plt.scatter(x[int(len(x)/2)], expert_values[characteristics_names[i]], s=100)
            plt.grid(True)
            plt.legend()
        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    with open('data/comparisons.pkl', 'rb') as f:
        comp_dataset = pickle.load(f)

    with open('data/expert_trajectories.pkl', 'rb') as f:
        expert_trajectories = pickle.load(f)

    print(comp_dataset[0])
    dataset_analyzer = DatasetAnalyzer(comp_dataset, expert_trajectories)
