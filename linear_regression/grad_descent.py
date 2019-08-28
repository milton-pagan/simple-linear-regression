import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class GradDescent(object):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


    def run(self, data_set):
        self.num_samples = len(data_set.index)

        features = data_set.to_numpy()

        y_values = features[:, -1]

        features = features[:, :-1]

        self.parameters = np.zeros(len(data_set.columns) - 1)

        temp_parameters = np.copy(self.parameters)

        test = []

        count = 0

        while(True):
            j = 0

            for param in self.parameters:
                summation = 0

                for feature_row, y_val in zip(features, y_values):
                    summation += ((np.matmul(self.parameters, feature_row)) - y_val) * feature_row[j]

                temp_parameters[j] = param - self.learning_rate * (1 / self.num_samples) * summation

                print(temp_parameters[j])

                j += 1



            prev_parameters = np.copy(self.parameters)

            self.parameters = np.copy(temp_parameters)

            print(self.parameters)

            test.append(self.cost_function(features, y_values))
            count += 1

            if self.check_convergence(prev_parameters):
                plt.plot(np.arange(count), test)
                plt.show()
                print(count)
                break



    def check_convergence(self, prev_parameters):

        for real_parameter, prev_parameter in zip(self.parameters, prev_parameters):
            if abs(real_parameter - prev_parameter) != 0:
                return False

        return True

    def cost_function(self, features, y_values):
        summation = 0

        for feature_row, y_val in zip(features, y_values):
            summation += ((np.matmul(self.parameters, feature_row)) - y_val) ** 2

        return (1 / (2 * self.num_samples)) * summation
