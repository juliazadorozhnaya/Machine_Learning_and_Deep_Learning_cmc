import numpy as np


class MinMaxScaler:
    def fit(self, data):
        data = np.array(data)
        self.minimum = []
        self.maximum = []
        for i in range(len(data[1])):
            self.minimum = np.min(data, axis=0)
            self.maximum = np.max(data, axis=0)

    def transform(self, data):
        return (data - self.minimum) / (self.maximum - self.minimum)


class StandardScaler:
    def fit(self, data):
        data = np.array(data)
        self.am = []
        for j in range(len(data[0])):
            self.am = np.mean(data, axis=0)

        self.disp = []
        for j in range(len(data[0])):
            self.disp = np.std(data, axis=0)

    def transform(self, data):
        return (data - self.am) / self.disp