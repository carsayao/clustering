# cmeans.py assumes data sits in "./data/rawdata/"

import numpy as np
import matplotlib.pyplot as plt
from read import Read
import sys
import os
import argparse

parser = argparse.ArgumentParser(description="K-means clustering")
parser.add_argument('--debug', action='store_true',
                    help="run with DEBUG flag set to True")
parser.add_argument('--plot', action='store_true',
                    help="plot the results")
args = parser.parse_args()
args = vars(args)
PLOT = args["plot"]


class Cmean:
    def __init__(self, f, start, end, k, m=2):
        # Number of features
        self.DEBUG = args["debug"]
        self.INPUTS = f
        self.SAMPLES = end-start
        self.start = start
        self.end = end
        # Number of clusters
        self.K = k
        # Fuzzifier
        self.M = m
        # Initialize our data
        self.data = self.load()

        # Constant centroids for debugging
        if self.DEBUG == True:
            self.K = 4
            # np.set_printoptions(threshold=sys.maxsize)
            self.centroids = np.array([[-2,-2],[-2,2],[2,2],[2,-2]])
        else:
            # Randomly initialize k centroids
            self.centroids = self.init_centroids()

        # Array to hold clusters of assigned datapoint
        # self.clusters = [np.zeros(2)] * self.K
        # Array to hold max grades
        self.predicts = np.zeros((self.SAMPLES, 2))
        # Array to hold membership grades
        # self.grades = np.zeros((self.SAMPLES, self.K))
        self.grades = np.random.randint(0, high=1, size=(self.SAMPLES, self.K))
    
    # Load self.SAMPLES number of datapoints with help from Read class
    def load(self):
        read = Read(self.INPUTS, self.start, self.end)
        # read = Read(self.INPUTS, self.SAMPLES)
        x = read.read_raw()
        return x

    # Initialize self.K number of centroids 
    def init_centroids(self):
        # Empty array of centroids to be filled, K long, INPUTS wide
        centroids = np.zeros((self.K, self.INPUTS))
        # Choose K sample indicies from our data without replacement.
        guesses = np.random.choice(self.data.shape[0],
                                   self.K, replace=False)
        # Grab our data points
        for k in range(self.K):
            centroids[k] = self.data[int(guesses[k])]
        # We will run into our randomly selected centroids when computing
        # weight updates. Calculating the norm will cause divide by zeros. 
        centroids += 0.00001
        return centroids
    
    # Use L2 norm: sqrt(sum(square(xi-yi)))
    # Calclulate weights for clusters
    # For each data point (x), generate arrays of Euclidean norm
    # to each centroid.
    def assignment(self):


        # Build norm_mat
        for i, x in enumerate(self.data):
            # Calculate norms
            norm = np.linalg.norm(x-self.centroids, axis=1)
            for j in range(self.K):
                for k in range(self.K):
                    self.grades[i][j] += (norm[j]/norm[k])**(2/(self.M-1))
            # import pdb; pdb.set_trace()
            # norm_mat[i] = norm
            # Get index of max grade
            self.grades[i] = 1/self.grades[i]
            index, = np.where(self.grades[i]>=np.amax(self.grades[i]))
            # print(i)
            # print(norm)
            # print(self.grades[i])
            # print(index)
            index = int(index)
            # print(self.grades[i][index])
            # Holding predictions array will help us color our plot
            self.predicts[i][0] = index
            self.predicts[i][1] = self.grades[i][index]
            # self.predicts[i][1] = np.amax(self.grades[i])
        
        i = 0
        # Use norm_mat to calculate weights
        # for i in range(self.data.shape[0]):
        #     for j in range(self.K):
        #         for k in range(self.K):
        #             self.grades[i][j] += (norm_mat[i][j]/norm_mat[i][k])**(2/(self.M-1))
                    # import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()


    
    # Update centroids
    def update(self):
        # Initialize new centroids
        new_centroids = np.zeros((self.K, self.INPUTS))
        # print("GRADES[:,0][:,np.newaxis]", self.grades[:,0][:,np.newaxis].shape)
        # print("data**m", (self.data**self.M).shape)
        # print("data", self.data.shape)

        # Perform update step on all centroids (slide 36 of lecture 9)
        for k in range(self.K):
        # for k in range(len(self.clusters)):
            dividend = np.sum((self.grades[:,k][:,np.newaxis] * (self.data**self.M) * self.data), axis=0)
            divisor  = np.sum((self.grades[:,k][:,np.newaxis] * (self.data**self.M)), axis=0)
            # print("dividend", dividend.shape)
            # print(dividend)
            # print("divisor", divisor.shape)
            # print(divisor)
            new_centroids[k] = dividend/divisor
            

            # new_centroids[k] = ((1/self.clusters[k].shape[0])
            #                  *  (np.sum(self.clusters[k], axis=0)))
        # Hold onto summed distance from new centroids to previous
        #   (used to evaluate stopping condition)
        d = np.linalg.norm(new_centroids-self.centroids, axis=1)
        d_sum = np.sum(d)
        # Update our centroids
        self.centroids = new_centroids
        # Reset clusters (because assignment() will keep stacking)
        # self.clusters = [np.zeros(2)] * self.K
        self.grades = np.zeros((self.SAMPLES, self.K))
        # Return summed distances so cmeans() can check for stopping condition
        return d_sum
        
    # Minimize within-cluster sum of squares by findiing loss of the r randomly
    #   initialized points (slide 12, lecture 9)
    def wcss(self):
        # Assign data points to clusters
        self.assignment()
        # Initialize wcss
        wcss = 0
        # For each cluster, sum the weight times the square of the distances
        # of each data point minus the centroid of the cluster and add that to wcss
        for k in range(self.K):
            norm = np.linalg.norm(self.data-self.centroids[k], axis=1)
            print("norm:",norm.shape)
            square = np.square(norm)
            wcss += np.sum(self.grades[:,k]*square)
            print("wcss:",wcss.shape)
        print("wcss", wcss)
        print()
        return wcss

    # Run cmean() until distance between new centroids and previous are less
    #   than stopping condition.
    def cmean(self, instance, stopping=0.0001):
        mode = 'C'
        print("*** cmean on instance %s ***" % instance)
        # Hold onto randomly initialized centroids to output them later
        initial_centroids = self.centroids
        # Hold our iteration number
        i = 0
        # Assign initial data points, plot, then update
        self.assignment()
        if PLOT == True:
            self.plot(instance, str(i), " m " + str(self.M) + " d= initialized", mode)
        centroid_distance = self.update()
        # Don't stop until we've reached our stopping condition
        while centroid_distance > stopping:
            print("D",centroid_distance)
            i += 1
            self.assignment()
            if PLOT == True:
                self.plot(instance, str(i), " m " + str(self.M) + " d=" + str(round(centroid_distance, 4)), mode)
            centroid_distance = self.update()
        i += 1
        if PLOT == True:
            self.plot(instance, str(i), " m " + str(self.M) + " d=" + str(round(centroid_distance, 4)), mode)

        # Get wcss value for instance
        wcss = self.wcss()
        return initial_centroids, wcss
        # return initial_centroids, 1

    # https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
    # https://stackoverflow.com/questions/31137077/how-to-make-a-scatter-plot-for-clustering-in-python
    def plot(self, instance, num, data, mode):
        # 4 modes of opacity

        # print("saving fig...")
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        title_inf = mode + "; K " + str(self.K) + "; instance " + instance + "; iteration " + num + "; " + data
        ax.set(xlabel='x', ylabel='y', title=title_inf)
        scatter = ax.scatter(self.data[:,0], self.data[:,1],
                             c=self.predicts[:,0], s=5)
        # scatter = ax.scatter(self.data[:,0], self.data[:,1],
        #                      c=self.predicts[:,0], s=5, alpha=self.predicts[:,1])
        plt.scatter(self.centroids[:,0],self.centroids[:,1],
                    c='black', s=100, alpha=0.5)
        plt.savefig("data/" + mode + "_" + str(self.K) + "_" + instance + "_" + num + ".png")
        plt.close(fig)
    

def main():
    # Features
    inputs = 2
    # Number of samples
    samples = 500
    start = 0
    end = 1500
    # Number of clusters
    clusters = 2
    m = 2
    # How many times to execute with randomly initialized centroids
    times = 2
    # Stopping condition
    stopping = .01
    # Hold list of initialized centroids of each k
    centroids = []
    # Hold array of wcss for each k
    wcss_list = []

    print(("\nRunning c-means with inputs=%s, start=%s, end=%s "
           "clusters=%s, m=%s, %s times, and stopping condition=%s\n"
      %
    (inputs, start, end, clusters, m, times, stopping)))

    # print(("\nRunning k-means with inputs=%s, samples=%s, "
    #        "clusters=%s, %s times, and stopping condition=%s\n"
    #   %
    # (inputs, samples, clusters, times, stopping)))

    # Run new instance of cmean according to params above
    for i in range(times):
        run = Cmean(inputs, start, end, clusters, m)
        # run = Kmean(inputs, samples, clusters)
        initial_centroids, wcss = run.cmean(str(i+1), stopping)
        centroids.append(initial_centroids)
        wcss_list.append(wcss)
    
    # Get index of our minimum wcss and print
    index_min = np.argmin(wcss_list)
    print("initialized points with minimum wcss: %s, wcss: %s"
      % (index_min, wcss_list[index_min]))
    print(centroids[index_min])

if __name__ == "__main__":
    main()