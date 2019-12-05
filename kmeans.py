# train.py assumes data sits in "./data/rawdata/"

import numpy as np
import matplotlib.pyplot as plt
from read import Read
import sys
import os
import argparse
# print((1/np.absolute(self.centroids[0].shape)).shape)

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

        if self.DEBUG == True:
            np.set_printoptions(threshold=sys.maxsize)
            self.centroids = np.array([[-2,-2],[-2,2],[2,2],[2,-2]])
            # zeros = np.zeros((self.K, self.INPUTS))
            # for i in range(self.TIMES):
                # self.centroids = np.vstack((self.centroids, zeros))
            # print(self.centroids)
        else:
            self.centroids = self.init_centroids()

        # Array to hold clusters
        self.clusters = [np.zeros(2)] * self.K
        # Array to hold predictions
        self.predicts = np.zeros(self.SAMPLES)
        self.title = None
        if self.DEBUG == True:
            print("\n*** init() ***")
    
    def load(self):
        read = Read(self.INPUTS, self.SAMPLES, debug=self.DEBUG)
        x = read.read_raw()
        return x

    def init_centroids(self):

        if self.DEBUG == True:
            print("\n*** init_centroids() ***")

        centroids = np.zeros((self.K, self.INPUTS))
        guesses = np.random.choice(self.data.shape[0], self.K, replace=False)
        
        if self.DEBUG == True:
            print("guesses: %s" % guesses)
            
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

        if self.DEBUG == True:
            print("\n*** assignment() ***")
            print("predicts: %s" % self.predicts.shape)

        # Hold index of cluster we are adding to
        i = 0
        # for x in range(self.data):
        for x in self.data:
            # Calculate norms
            norm = np.linalg.norm(x-self.centroids, axis=1)
            # Get index of min val
            index, = np.where(norm<=np.amin(norm))
            # Convert to int to use as index
            # import pdb; pdb.set_trace()
            index = int(index)

            self.predicts[i] = index

            # If the cluster at index is empty, set to x
            if (self.clusters[index]==0).any():
                self.clusters[index] = x
            # Else, we just stack x at index
            else:
                self.clusters[index] = np.vstack((self.clusters[index], x))
            
            i += 1

        # Check that we are using all our data
        if self.DEBUG == True:
            total = 0
            for c in self.clusters:
                total += c.shape[0]
                print(c.shape)
            print("total in clusters:", total)
            print("data:", self.data.shape)
            print(self.predicts[:self.K])
    
    def update(self):

        if self.DEBUG == True:
            print("\n*** update() ***")

        new_centroids = np.zeros((self.K, self.INPUTS))
        for k in range(len(self.clusters)):
            # new_centroids[k] = np.sum(self.clusters[k], axis=0)
            if self.DEBUG == True:
                print("new_centroids[%s].shape:    " % k, new_centroids.shape)
                print("self.clusters[k].shape[%s]):" % k, self.clusters[k].shape[0])
                print("1/self.clusters[%s].shape:  " % k, (1/(self.clusters[k].shape[0])))
                print("sum(cluster[%s]).shape:     " % k, (np.sum(self.clusters[k], axis=0).shape))
            new_centroids[k] = (1/self.clusters[k].shape[0])*np.sum(self.clusters[k], axis=0)
        d = np.linalg.norm(new_centroids-self.centroids, axis=1)        
        d_sum = np.sum(d)

        if self.DEBUG == True:
            print()
            print("new_centroids")
            print(new_centroids)
            print(new_centroids.shape)
            print("d")
            print(d)
            print("sum(d)")
            print(d_sum)
        else:
            print("d")
            print(d)
            print("sum(d)")
            print(d_sum)

        self.centroids = new_centroids
        self.clusters = [np.zeros(2)] * self.K
        return d_sum
        
    # Slide 12
    # Minimize within-cluster sum of squares by findiing loss of the r randomly
    # initialized points.
    def wcss(self):

        if self.DEBUG == True:
            print("\n*** wcss() ***")

        for x in self.centroids:
            print("wcss",x)
            print("wcss",x.shape)

    def kmean(self, save_name, stopping=0.2):

        if self.DEBUG == True:
            print("\n*** kmean() ***")

        i = 0
        self.assignment()
        self.plot(save_name, "GMM" + str(i), '')
        centroid_distance = self.update()
        while centroid_distance >= stopping:
            i += 1
            self.assignment()
            self.plot(save_name, "GMM" + str(i), "_" + str(centroid_distance))
            centroid_distance = self.update()
        i += 1
        self.plot(save_name, "GMM" + str(i), "_" + str(centroid_distance))
    
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
    # https://stackoverflow.com/questions/31137077/how-to-make-a-scatter-plot-for-clustering-in-python
    def plot(self, save_name, title, data):
        fig = plt.figure(figsize=(7,6))
        # fig = plt.figure()
        # title_info = (self.)
        ax = fig.add_subplot(111)
        ax.set(xlabel='x', ylabel='y', title=title+data)
        scatter = ax.scatter(self.data[:,0], self.data[:,1], c=self.predicts, s=5)
        plt.scatter(self.centroids[:,0],self.centroids[:,1], c='black', s=100, alpha=0.5)
        plt.savefig("data/" + save_name + "_" + title + ".png")
    
    def test(self):
        print("\n*** test() ***")
        print("\nself.centroids:\n%s" % self.centroids)

def main():
    inputs = 2
    samples = 1500
    clusters = 4
    times = 10

    for i in range(times):
        run = Kmean(inputs, samples, clusters, times)
        if DEBUG == True:
            run.test()
        run.kmean(str(i+1))

    # for i in range(times):
    #     run.assignment()
    #     run.plot("GMM" + str(i))
    #     run.wcss()
    #     run.update()

if __name__ == "__main__":
    main()
