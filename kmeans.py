# train.py assumes data sits in "./data/rawdata/"

import numpy as np
from read import Read
import sys
import os
import argparse
# np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser(description="K-means clustering")
parser.add_argument('--debug', action='store_true',
                    help="run with DEBUG flag set to True")
args = parser.parse_args()
args = vars(args)
DEBUG = args["debug"]

class Kmean:
    def __init__(self, f, n, k, r):
        self.INPUTS = f
        self.SAMPLES = n
        self.K = k
        self.TIMES = r
        self.DEBUG = args["debug"]
        self.data = self.load()
        self.centroids = self.init_centroids()
        self.clusters = [np.zeros(2)] * self.K
    
    def load(self):
        read = Read(self.INPUTS, self.SAMPLES, debug=self.DEBUG)
        x = read.read_raw()
        return x

    def init_centroids(self):
        centroids = np.zeros((self.K, self.INPUTS))
        guesses = np.random.choice(self.data.shape[0], self.K, replace=False)
        if self.DEBUG == True:
            print("\nguesses: %s" % guesses)
        for k in range(self.K):
            centroids[k] = self.data[int(guesses[k])]
            if self.DEBUG == True:
                print("k %s" % k)
                print("guesses[k]: %s" % int(guesses[k]))
                print(self.data[int(guesses[k])])
        return centroids
    
    # Use L2 norm: sqrt(sum(square(xi-yi)))
    # For each data point (x), generate arrays of Euclidean norm
    # to each centroid. Get index of min val, then add x to cluster
    # array at that index.
    def assignment(self):
        for x in self.data:
            # Calculate norms
            norm = np.linalg.norm(x-self.centroids, axis=1)
            # Get index of min val
            index, = np.where(norm<=np.amin(norm))
            # Convert to int to use as index
            index = int(index)
            # If the cluster at index is empty, set to x
            if (self.clusters[index]==0).any():
                self.clusters[index] = x
            # Else, we just stack x at index
            else:
                self.clusters[index] = np.vstack((self.clusters[index], x))
        # Check that we are using all our data
        if self.DEBUG == True:
            total = 0
            for c in self.clusters:
                total += c.shape[0]
                print(c.shape)
            print(total)
    
    def test(self):
        print("\nself.centroids:\n%s" % self.centroids)

def main():
    inputs = 2
    samples = 1500
    clusters = 3
    times = 10

    run = Kmean(inputs, samples, clusters, times)
    if DEBUG == True:
        run.test()
    run.assignment()

if __name__ == "__main__":
    main()