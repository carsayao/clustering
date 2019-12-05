# kmeans.py assumes data sits in "./data/rawdata/"

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
parser.add_argument('--plot', action='store_true',
                    help="plot the results")
args = parser.parse_args()
args = vars(args)
DEBUG = args["debug"]
PLOT = args["plot"]

class Kmean:
    def __init__(self, f, n, k):
        self.DEBUG = args["debug"]
        # Number of features
        self.INPUTS = f
        self.SAMPLES = n
        # Number of clusters
        self.K = k
        # Initialize our data
        self.data = self.load()

        # Constant centroids for debugging
        if self.DEBUG == True:
            self.K = 4
            np.set_printoptions(threshold=sys.maxsize)
            self.centroids = np.array([[-2,-2],[-2,2],[2,2],[2,-2]])
        else:
            # Randomly initialize k centroids
            self.centroids = self.init_centroids()

        # Array to hold clusters of assigned datapoint
        self.clusters = [np.zeros(2)] * self.K
        # Array to hold predictions
        self.predicts = np.zeros(self.SAMPLES)
        if self.DEBUG == True:
            print("\n*** init() ***")
    
    # Load self.SAMPLES number of datapoints with help from Read class
    def load(self):
        read = Read(self.INPUTS, self.SAMPLES, debug=self.DEBUG)
        x = read.read_raw()
        return x

    # Initialize self.K number of centroids 
    def init_centroids(self):
        if self.DEBUG == True:
            print("\n*** init_centroids() ***")
        # Empty array of centroids to be filled, K long, INPUTS wide
        centroids = np.zeros((self.K, self.INPUTS))
        # Choose K sample indicies from our data without replacement.
        guesses = np.random.choice(self.data.shape[0], self.K, replace=False)
        
        if self.DEBUG == True:
            print("guesses: %s" % guesses)
            
        # Grab our data points
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

        # for x in range(self.data):
        for i, x in enumerate(self.data):
            # Calculate norms
            norm = np.linalg.norm(x-self.centroids, axis=1)
            # Get index of min val
            index, = np.where(norm<=np.amin(norm))
            # Convert to int to use as index
            # import pdb; pdb.set_trace()
            index = int(index)

            # Holding predictions array will help us color our plot
            self.predicts[i] = index

            # If the cluster at index is empty, set to x
            # (Will only apply to first few calls to assignment (hopefully!))
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
            print("total in clusters:", total)
            print("data:", self.data.shape)
            print(self.predicts[:self.K])
    
    # Update centroids
    def update(self):

        if self.DEBUG == True:
            print("\n*** update() ***")

        # Initialize new centroids
        new_centroids = np.zeros((self.K, self.INPUTS))
        # Perform update step on all centroids (slide 15 of lecture 9)
        for k in range(len(self.clusters)):
            if self.DEBUG == True:
                print("new_centroids[%s].shape:    " % k, new_centroids.shape)
                print("self.clusters[k].shape[%s]):" % k, self.clusters[k].shape[0])
                print("1/self.clusters[%s].shape:  " % k, (1/(self.clusters[k].shape[0])))
                print("sum(cluster[%s]).shape:     " % k, (np.sum(self.clusters[k], axis=0).shape))
            new_centroids[k] = ((1/self.clusters[k].shape[0])
                             *  (np.sum(self.clusters[k], axis=0)))
        # Hold onto summed distance from new centroids to previous
        #   (used to evaluate stopping condition)
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
        # else:
        #     print("\nd",d)
        #     print("sum(d)",d_sum)

        # Update our centroids
        self.centroids = new_centroids
        # Reset clusters (because assignment() will keep stacking)
        self.clusters = [np.zeros(2)] * self.K
        # Return summed distances so kmeans() can check for stopping condition
        return d_sum
        
    # Minimize within-cluster sum of squares by findiing loss of the r randomly
    #   initialized points (slide 12, lecture 9)
    def wcss(self):

        if self.DEBUG == True:
            print("\n*** wcss() ***")
        
        # Assign data points to clusters
        self.assignment()
        # Initialize wcss
        wcss = 0

        # For each cluster, sum the square of the distances of each data
        #  point minus the mean of the cluster and add that to wcss
        for i, x in enumerate(self.clusters):
            norm = np.linalg.norm(x-self.centroids[i], axis=1)
            if self.DEBUG == True:
                print("cluster[%s]: %s" % (i, x.shape))
                print("norm: %s" % norm.shape)
            square = np.square(norm)
            wcss += np.sum(square)
        print("wcss", wcss)
        print()
        return wcss

    # Run kmean until distance between new centroids and previous are less
    #   than stopping condition.
    def kmean(self, instance, stopping=0.2):
        # Hold onto randomly initialized centroids to output them later
        initial_centroids = self.centroids
        # Hold our iteration number
        i = 0
        # Assign data points
        self.assignment()
        if PLOT == True:
            self.plot(instance, str(i), " initialized")
        print("*** kmean on instance %s ***" % instance)
        centroid_distance = self.update()
        while centroid_distance > stopping:
            i += 1
            self.assignment()
            if PLOT == True:
                self.plot(instance, str(i), " d=" + str(round(centroid_distance, 4)))
            centroid_distance = self.update()
        i += 1
        if PLOT == True:
            self.plot(instance, str(i), " d=" + str(round(centroid_distance, 4)))

        # Get wcss value for instance
        wcss = self.wcss()
        return initial_centroids, wcss

    # https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
    # https://stackoverflow.com/questions/31137077/how-to-make-a-scatter-plot-for-clustering-in-python
    def plot(self, instance, num, data):
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        title_inf = "K " + str(self.K) + "; instance " + instance + "; iteration " + num + "; " + data
        ax.set(xlabel='x', ylabel='y', title=title_inf)
        scatter = ax.scatter(self.data[:,0], self.data[:,1],
                             c=self.predicts, s=5)
        plt.scatter(self.centroids[:,0],self.centroids[:,1],
                    c='black', s=100, alpha=0.5)
        plt.savefig("data/" + str(self.K) + "_" + instance + "_" + num + ".png")
        plt.close(fig)
    
def main():
    # Features
    inputs = 2
    # Number of samples
    samples = 1500
    # Number of clusters
    clusters = 12
    # How many times to execute with randomly initialized centroids
    times = 10
    # Stopping condition
    stopping = 0
    # Hold list of initialized centroids of each k
    centroids = []
    # Hold array of wcss for each k
    wcss_list = []

    print("\nRunning k-means with inputs=%s, samples=%s, clusters=%s, %s times, and stopping condition=%s\n"
      %
    (inputs, samples, clusters, times, stopping))

    # Run new instance of kmean according to params above
    for i in range(times):
        run = Kmean(inputs, samples, clusters)
        initial_centroids, wcss = run.kmean(str(i+1), stopping)
        centroids.append(initial_centroids)
        wcss_list.append(wcss)
    
    # Get index of our minimum wcss and print
    index_min = np.argmin(wcss_list)
    print("initialized points with minimum wcss")
    print(centroids[index_min])

if __name__ == "__main__":
    main()