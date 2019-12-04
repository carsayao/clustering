import numpy as np
import os
import sys
from read import Read
# from random import choices
# from random import random
import random

arg = sys.argv[1]

if arg == 'random':
    # a = [[0,0],
    #      [0,1],
    #      [1,0],
    #      [1,1]]
    a = np.array([[0,0,0],
                  [0,0,1],
                  [0,1,0],
                  [0,1,1],
                  [1,0,0],
                  [1,0,1],
                  [1,1,0],
                  [1,1,1]])
    print("a: %sx%s" % a.shape)
    s = np.random.choice(a.shape[0], 3, replace=False)
    # s = np.random.choice(a.shape[0], 3, replace=True)
    # s = random.choices(a, k=3)
    print("s: %s" % s)

if arg == 'scope':
    class Myclass:
        def __init__(self):
            self.centroid = self.set_centroid()
        def set_centroid(self):
            centroid = 5
            return centroid
        def print_centroid(self):
            print("self.centroid: %s" % self.centroid)
    instance = Myclass()
    instance.set_centroid()
    instance.print_centroid()

if arg == 'norm':
    data = np.array([[-4,  32],
                     [3,   5.2]])
                    #  [1.4, 5]])
    centroids = np.array([[1,   .5],
                          [-7.1, 2.77]])
    # diff = 
    # square = np.square(diff)
    # sums = np.sum(square, axis=1)
    # root = np.sqrt(sums)
    norm = np.linalg.norm(data-centroids, axis=1)
    print("data")
    print(data)
    print("centroids")
    print(centroids)
    # print("diff")
    # print(diff)
    # print("square")
    # print(square)
    # print("square.shape")
    # print(square.shape)
    # print("sums")
    # print(sums)
    # print("root")
    # print(root)
    print("norm")
    print(norm)
    # norm = np.sqrt()

if arg == 'clusters':
    data = np.array([[-4,  32],
                     [3,   5.2],
                     [1.4, 5],
                     [3.2, 6.1],
                    ])
    centroids = np.array([[1,   .5],
                          [-7.1, 2.77],
                          [-.21, 13],
                        ])
    # clusters = [None] * centroids.shape[0]
    clusters = [np.zeros(2)] * centroids.shape[0]
    # clusters = [0] * centroids.shape[0]
    # clusters = [0,1,2]
    print(clusters)
    for x in data:
        print(x)
        # norm = np.zeros((centroids.shape[0]))
        norm = np.linalg.norm(x-centroids, axis=1)
        print(norm)
        index, = np.where(norm<=np.amin(norm))
        index = int(index)
        print(index)
        if (clusters[index]==0).any():
            clusters[index] = x
        else:
            clusters[index] = np.vstack((clusters[index], x))

        # clusters[index] = np.vstack((clusters[index], x))

        # clusters[index] = np.append(clusters[index], x)
        # print(clusters[int(index)])
        
    # Add data to empty cluster[1]
    # clusters[1] = np.vstack((clusters[1], [1,1]))
    # Try deleting the [0,0]'s from clusters
    # clusters = np.delete(clusters, (0), axis=0)

    # clusters[0] = np.array([3,3])
    # clusters[0] = np.vstack((clusters[0], [1,1]))
    print("clusters")
    print(clusters)
    print()
    print("clusters[0]")
    print(clusters[0])
    print()

    x = np.array([[0,0],
                  [0,1]])
    x = np.vstack((x, [1,0]))
    print("x")
    print(x)
    print()

if arg == 'read':
    obj = Read(2, 1500)
    data = obj.read_raw()
    print((data==-.1695138).any())