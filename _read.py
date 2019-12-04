# read.py assumes data sits in "./data/rawdata/"

import numpy as np
import sys
import os


class Read:

    # def __init__(self, inputs, samples):
    def __init__(self, inputs, samples, debug=False):
        self.INPUTS = inputs
        self.SAMPLES = samples
        self.DEBUG = debug
        # File paths
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.data_raw = self.path + "/data/rawdata/GMM_data_fall2019.txt"
        # self.data_dat = self.path + "/data/data.dat"
        self.train_dat = self.path + "/data/train.dat"

    # Read raw data from GMM_data_fall2019.txt, return set
    # train_set.shape: (1500,2)
    def read_raw(self):
        # Read data into x
        print("reading raw data...")

        # train_set = np.loadtxt(fname=self.data_raw)
        # with open('data/rawdata/GMM_data_fall2019.txt', 'r') as f:
            # train_set = f.readlines()
        f = open('data/rawdata/GMM_data_fall2019.txt')
        train_set = f.read()

        # Deep copy data
        # train_set = x.copy()
        # if self.DEBUG == True:
        #     print(train_set.shape)
        return train_set

def main():
    inputs = 2
    samples = 1500

    # import pdb; pdb.set_trace()
    run = Read(inputs, samples, debug=True)
    x = run.read_raw()
    # print(x[:5])
    print(x[0])
    print(len(x))

if __name__ == "__main__":
    main()