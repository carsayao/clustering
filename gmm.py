import os
import re
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from read import Read


parser = argparse.ArgumentParser(description="Gaussian Mixture Modelling")
parser.add_argument('--plot', action='store_true',
                    help="plot the results")
parser.add_argument('--debug', action='store_true',
                    help="set trace with pdb")
parser.add_argument('--test', action='store_true',
                    help="test function from test.py")
args = parser.parse_args()
args = vars(args)
PLOT = args["plot"]
if args["debug"] == True:
    import pdb; pdb.set_trace()


class GMM:
    def __init__(self, mean_path, sigma_path):
        """Initialialize parameters mean and covariance randomly.
        
        Random means, sigmas. Priors equally distributed where sum(priors) = 1.
        """
        self.plotpath = "./data/plots/"
        self.means = np.loadtxt(fname=mean_path)
        self.k = self.means.shape[0]
        # Deep copy from memmap
        fp = np.memmap(sigma_path, dtype='float64', mode='r',
                       shape=(self.means.shape[0],2,2))
        self.covariances = np.empty((np.shape(fp)))
        self.covariances[:] = fp[:]
        
        self.priors = np.ones((self.k)) / self.k
        self.data = np.loadtxt(fname="./data/rawdata/GMM_dataset_546.txt")
    
    
    def gauss(self, mean, cov, x) -> (1500, 2):
        """Multivariate Gaussian
        Args:
            mean (np.array) : shape (2,)
            cov (np.array) : shape (2,2)
            x (np.array) : shape (N,2)

            (x-mean).shape==(1500,2)
            np.dot(x-mean,np.linalg.inv(cov))==(1500,2)
        """
        d = self.means.shape[1]
        numer = []
        denom = []
        for n in range(self.data.shape[0]):
            numer.append(np.exp(-.5 * ( np.linalg.solve(cov,(x[n]-mean)).T.dot(x[n]-mean) )))
            denom.append((2 * np.pi)**(d/2) * (np.linalg.det(cov)**.5))
        numer = np.array(numer)
        denom = np.array(denom)

        return numer/denom

    
    def gamma(self, prior, mean, cov, x) -> (1500, 2):
        """Gamma
        Args:
            prior (float) : shape ()
            mean (np.array) : shape (2,)
        """
        return prior * self.gauss(mean, cov, x)
    
    def gmm(self):
        """GMM algorithm
        Args:
        
        """
        for r in range(50):
            # Expectation
            conditionals = []
            for j in range(self.k):
                cond = self.gamma(self.priors[j], self.means[j], self.covariances[j], self.data)
                conditionals.append(cond)
            conditionals = np.array(conditionals)
            # Posteriors
            posteriors = []
            for j in range(self.k):
                posteriors.append(np.log(conditionals[j] / np.sum(conditionals)))
            # posteriors.shape==(k,n)
            posteriors = np.array(posteriors)
            # Get assignments
            z = np.zeros((posteriors.T.shape))
            # Predictions array for plot
            predicts = np.zeros((self.data.shape[0]))
            for n in range(self.data.shape[0]):
                # Get one hots
                z[n] = np.where(posteriors.T[n]>=np.amax(posteriors.T[n]), 1,0)
                # Extract index from layers of arrays
                predicts[n] = np.where(z[n]>=np.amax(z[n]))[0][0]
            nk = np.sum(posteriors, axis=1)

            self.plot(self.means, predicts, r)

            # TODO: Maximization
            print(self.means)
            for j in range(self.k):
                self.means[j] = (1/nk[j]) * np.sum(posteriors[j].reshape(self.data.shape[0],1) * self.data)
                first = posteriors[j]
                # print(f"first {first.shape}")
                second = (self.data-self.means[j]).T
                # print(f"second {second.shape}")
                third = (self.data-self.means[j])
                # print(f"third {third.shape}")
                self.covariances[j] = (1/nk[j])*np.dot((first*second), third)

                self.priors[j] = nk[j]/self.data.shape[0]
            print(self.means.shape)
            print(self.means)
            print(self.covariances)
            print(self.covariances.shape)
            print(f"priors {np.sum(self.priors)}")

    def plot(self, means, predicts, instance):
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        title_inf = ("G " + str(self.k)
                   + "; instance " + str(instance))
        ax.set(xlabel='x', ylabel='y', title=title_inf)
        scatter = ax.scatter(self.data[:,0], self.data[:,1],
                             c=predicts, s=5)
        plt.scatter(means[:,0], means[:,1],
                    c=np.array(np.arange(self.k)), s=100)
        plt.savefig(f"./data/plots/G{str(self.k)}_{str(instance)}")
        plt.close()


def main():
    for i in range(2,11):
        gmm = GMM(f"./data/K-{str(i)}_means.txt", f"./data/K-{str(i)}_cov.dat")
        gmm.gmm()


if __name__ == "__main__":
    main()
