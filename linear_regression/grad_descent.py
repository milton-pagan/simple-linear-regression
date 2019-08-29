import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class GradDescent(object):

    def __init__(self, data_set, learning_rate):
        self.learning_rate = learning_rate

        self.num_samples = len(data_set.index)

        self.features = data_set.to_numpy()

        self.y_values = self.features[:, -1]

        self.features = self.features[:, :-1]

        self.parameters = np.zeros(len(data_set.columns) - 1)

    def run(self):

        temp_parameters = np.copy(self.parameters)

        costs = []

        count = 0

        while(True):
            j = 0

            for param in self.parameters:
                summation = 0

                for feature_row, y_val in zip(self.features, self.y_values):
                    summation += ((np.matmul(self.parameters, feature_row)) - y_val) * feature_row[j]

                temp_parameters[j] = param - self.learning_rate * (1 / self.num_samples) * summation

                j += 1

            self.parameters = np.copy(temp_parameters)

            costs.append(self.cost_function())
            print(costs[count])
            count += 1

            if count > 2 and self.check_convergence(*costs):
                plt.plot(np.arange(count), costs)
                plt.show()
                print(count)
                break

    def check_convergence(self, *costs):
        return costs[-1] == costs[-2]

    def cost_function(self):
        summation = 0

        for feature_row, y_val in zip(self.features, self.y_values):
            summation += ((np.matmul(self.parameters, feature_row)) - y_val) ** 2

        return (1 / (2 * self.num_samples)) * summation

    def change_dataset(self, data_set):

        self.num_samples = len(data_set.index)

        self.features = data_set.to_numpy()

        self.y_values = self.features[:, -1]

        self.features = self.features[:, :-1]

        self.parameters = np.zeros(len(data_set.columns) - 1)

    def mean_normalize(self):

        for column in self.features.transpose():
            mean = column.mean()
            std = column.std()

            for i in range(len(column)):
                if std != 0:
                    column[i] = (column[i] - mean) / std

        mean = self.y_values.mean()
        std = self.y_values.std()

        for i in range(len(self.y_values)):
            if std != 0:
                self.y_values[i] = (self.y_values[i] - mean) / std
