# read.py assumes data sits in "./data/rawdata/"

import numpy as np
import sys
import os

class Read:
    def __init__(self, inputs, start, end):
    # def __init__(self, inputs, samples):
        self.INPUTS = inputs
        self.SAMPLES = end-start
        self.start = start
        self.end = end
        # self.SAMPLES = samples
        # File paths
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.data_raw = self.path + "/data/rawdata/GMM_data_fall2019.txt"
        self.train_dat = self.path + "/data/train.dat"

    # Read raw data from GMM_data_fall2019.txt, return set
    def read_raw(self):
        # Read data into x
        train_set = np.loadtxt(fname=self.data_raw)
        return train_set[self.start:self.end]
        # return train_set[:self.SAMPLES]