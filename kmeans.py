# kmeans.py assumes data sits in "./data/rawdata/"
# Main idea here is: 
# For every data point, assign it to a cluster. Then:
# For every cluster, stack up all the data points that
# belong to the cluster and update centroids.

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from read import Read

parser = argparse.ArgumentParser(description="K-means clustering")
parser.add_argument('--plot', action='store_true',
                    help="plot the results")
args = parser.parse_args()
args = vars(args)
PLOT = args["plot"]


class Kmean:
    def __init__(self, f, start, end, k, path="./data/"):
        # Number of features
        self.INPUTS = f
        self.SAMPLES = end-start
        self.start = start
        self.end = end
        # Number of clusters
        self.K = k
        # Initialize our data
        self.data = self.load()
        # Randomly initialize k centroids
        self.centroids = self.init_centroids()
        # Array to hold clusters of assigned datapoint
        self.clusters = [np.zeros(2)] * self.K
        # Array to hold predictions
        self.predicts = np.zeros(self.SAMPLES)
        # Dir to hold plots
        self.dir = path
    
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
        return centroids
    
    # Use L2 norm: sqrt(sum(square(xi-yi)))
    # For each data point (x), generate arrays of Euclidean norm
    # to each centroid. Get index of min val, then add x to cluster
    # array at that index.
    def assignment(self):
        for i, x in enumerate(self.data):
            # Calculate norms
            norm = np.linalg.norm(x-self.centroids, axis=1)
            # Get index of min val
            index, = np.where(norm<=np.amin(norm))
            # Convert to int to use as index
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
        # import pdb; pdb.set_trace()
        # for i in range(self.clusters.shape[0]):
        #     print(self.clusters[i].shape)
    
    # Update centroids
    def update(self):
        # Initialize new centroids
        new_centroids = np.zeros((self.K, self.INPUTS))
        # Perform update step on all centroids (slide 15 of lecture 9)
        for k in range(len(self.clusters)):
            new_centroids[k] = ((1/self.clusters[k].shape[0])
                              * (np.sum(self.clusters[k], axis=0)))
        # Hold onto summed distance from new centroids to previous
        #   (used to evaluate stopping condition)
        d = np.linalg.norm(new_centroids-self.centroids, axis=1)
        d_sum = np.sum(d)
        # Update our centroids
        self.centroids = new_centroids
        # Reset clusters (because assignment() will keep stacking)
        self.clusters = [np.zeros(2)] * self.K
        # Return summed distances so kmeans() can check for stopping condition
        return d_sum
        
    # Minimize within-cluster sum of squares by findiing loss of the r randomly
    #   initialized points (slide 12, lecture 9)
    def wcss(self):
        # Assign data points to clusters
        self.assignment()
        # Initialize wcss
        wcss = 0
        # For each cluster, sum the square of the distances of each data
        #  point minus the mean of the cluster and add that to wcss
        for i, x in enumerate(self.clusters):
            norm = np.linalg.norm(x-self.centroids[i], axis=1)
            square = np.square(norm)
            wcss += np.sum(square)
        return wcss
    
    def covariance(self):
        cov = []
        for j, x in enumerate(self.clusters):
            cov.append(np.cov(x.T))
        return cov

    # Run kmean() until distance between new centroids and previous are less
    #   than stopping condition.
    def kmean(self, instance, stopping=0.2):
        # Hold onto randomly initialized centroids to output them later
        initial_centroids = self.centroids
        # Hold our iteration number
        i = 0
        # Assign initial data points, plot, then update
        self.assignment()
        if PLOT == True:
            self.plot(instance, str(i), " initialized")
        print("* kmean on instance %s *" % instance)
        centroid_distance = self.update()
        # Don't stop until we've reached our stopping condition
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
        # Get covariance
        cov = self.covariance()
        print(f"wcss {wcss}")
        # print(f"cov\n{cov}")
        return initial_centroids, self.centroids, wcss, cov

    # https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
    # https://stackoverflow.com/questions/31137077/how-to-make-a-scatter-plot-for-clustering-in-python
    def plot(self, instance, num, data):
        print("saving fig...")
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        title_inf = ("K " + str(self.K)
                   + "; instance " + instance
                   + "; iteration " + num
                   + "; " + data)
        ax.set(xlabel='x', ylabel='y', title=title_inf)
        scatter = ax.scatter(self.data[:,0], self.data[:,1],
                             c=self.predicts, s=5)
        plt.scatter(self.centroids[:,0],self.centroids[:,1],
                    c='black', s=100, alpha=0.5)
        # plt.show()
        plt.savefig(self.dir + "K" + str(self.K) + "_"
                  + instance + "_" + num + ".png")
        plt.close(fig)

    
def run_kmeans(times, stopping, inputs, start, end, clusters, save_dir):
    # for i in range(1,clusters+1):
    for i in range(2, clusters+1):

        wcss_min = 1e4
        centroid_min = []
        end_centroid_min = []
        cov_min = []

        print(("\nRunning k-means with inputs=%s, start=%s, end=%s " \
            "clusters=%s, %s times, and stopping condition=%s\n"
            % (inputs, start, end, i, times, stopping)))

        # Run new instance of kmean according to params above
        for j in range(1, times+1):
            run = Kmean(inputs, start, end, i, path=save_dir)
            initial_centroids, end_centroids, wcss, cov = run.kmean(str(j), stopping)
            if wcss <= wcss_min:
                wcss_min = wcss
                centroid_min = initial_centroids
                end_centroid_min = end_centroids
                cov_min = cov
        
        print(f"initialized points with minimum wcss: {wcss_min} \
               \ncentroid_min\n{centroid_min} \
               \nend_centroids\n{end_centroid_min} \
               \ncov_min\n{cov_min}")
               
        # print(f"\nMinimum wcss of all: {wcss_min}, with centroids:")
        # print(centroid_min)
        # print(f"And covariance:\n{cov_min}")

        title = f"{'./data/'}K-{centroid_min.shape[0]}"
        wc = f"{int(wcss_min)}"
        np.savetxt(f"{title}_means.txt",
                end_centroid_min)
        fp = np.memmap(f"{title}_cov.dat", dtype='float64', mode='w+', shape=(len(cov_min),2,2))
        fp[:] = cov_min[:]
        del fp
        with open(f"{title}_{wc}_cov.txt", 'w') as f:
            f.write('\n')
            for m in cov_min:
                np.savetxt(f, m, fmt='%-10.5f')
                f.write('\n')
        # wcss_min = 1e4
        # centroid_min = []
        # cov_min = []

def once(stopping, inputs, start, end, clusters, save_dir):
    run = Kmean(inputs, start, end, clusters, path=save_dir)
    initial_centroids, wcss, cov = run.kmean("1", stopping)
    print(f"wcss{wcss}init_centroids\n{initial_centroids}\n \
            covariance\n{cov}\n")

def main():
    # Features
    inputs = 2
    # Number of samples
    samples = 500
    start = 0
    end = 1500
    # Number of clusters
    clusters = 10
    # How many times to execute with randomly initialized centroids
    times = 5
    # Stopping condition
    stopping = 0

    np.set_printoptions(precision=3)

    save_dir = "./data/plots/"

    try:
        Path(save_dir).mkdir()
        print(f"[+] {save_dir} created.")
    except FileNotFoundError:
        save_dir = "./data/plots/"
        print(f"[!] {save_dir}'s parent directory does not exist! \
                Defaulting to {save_dir}.'")
        try:
            Path(save_dir).mkdir(parents=True)
            print(f"[+] {save_dir} created.")
        except FileExistsError:
            print(f"[!] {save_dir} already exists!")
    except FileExistsError:
        print(f"[+] {save_dir} already exists.")
    
    # once(stopping, inputs, start, end, clusters, save_dir)
    run_kmeans(times, stopping, inputs, start, end, clusters, save_dir)


if __name__ == "__main__":
    main()