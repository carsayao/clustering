import numpy as np
import matplotlib.pyplot as plt
from read import Read
import os
import sys
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
    mean = np.mean(data, axis=0)
    # diff = data-centroids
    diff = data-mean
    square = np.square(diff)
    sums = np.sum(square, axis=1)
    root = np.sqrt(sums)
    print("mean",mean)
    print("data-mean",diff)
    norm = np.linalg.norm(data-mean, axis=1)
    # norm = np.linalg.norm(data-centroids, axis=1)
    print("data")
    print(data)
    print("mean")
    print(mean)
    # print("centroids")
    # print(centroids)
    # print("diff")
    # print(diff)
    # print("square")
    # print(square)
    # print("square.shape")
    # print(square.shape)
    # print("sums")
    # print(sums)
    print("root")
    print(root)
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
    # print((data==-.1695138).any())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    predicts = np.array([0,1,2,3])
    scatter = ax.scatter(data[:,0], data[:,1], s=5)
    plt.savefig("test_scatter_GMM.png")

if arg == 'plot':
    data      = np.array([[ -4  , 3.4 ],
                          [  3  , 5.2 ],
                          [  1.4, 5   ],
                          [  3.2, 6.1 ],
                        ])
    centroids = np.array([[  1   ,   .5 ],
                          [ -7.1 ,  2.7 ],
                          [  -.21, 13   ],
                        ])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = data[:,0]
    y = data[:,1]
    print("x")
    print(x)
    print(x.shape)
    print("y")
    print(y)
    print(y.shape)
    scatter = ax.scatter(x,y)
    plt.savefig("test_scatter.png")

if arg == 'sum':

    a = np.array([[0,0,0],
                  [0,0,1],
                  [0,1,0],
                  [0,1,1],
                  [1,0,0],
                  [1,0,1],
                  [1,1,0],
                  [1,1,1]])
    print("a: %sx%s" % a.shape)
    print(np.sum(a, axis=0))
    print(np.sum(a, axis=1))

if arg == 'tuple':
    a = np.array([[0,0],
                  [0,0],
                  [0,1],
                  [0,1]])
    b = np.array([[1,0],
                  [1,0],
                  [1,1],
                  [1,1]])
    centroids = []
    # a = np.array([(0,0),
    #               (0,0),
    #               (0,1),
    #               (0,1)])
    # b = np.array([(1,0),
    #               (1,0),
    #               (1,1),
    #               (1,1)])
    print(a)
    print()
    print(b)
    print()
    centroids.append(a)
    # print(centroids)
    centroids.append(b)
    print(centroids)
    print()
    print(centroids[0])
    print()
    print(centroids[1])
    # import pdb; pdb.set_trace()
    # a = np.append(a, b, axis=0)
    # a = np.vstack((a, b))
    # a = np.concatenate(a, b)

if arg == 'var':
    data      = np.array([[ -4  , 3.4 ],
                          [  3  , 5.2 ],
                          [  1.4, 5   ],
                          [  3.2, 6.1 ],
                        ])
    centroids = np.array([[  1   ,   .5 ],
                          [ -7.1 ,  2.7 ],
                          [  -.21, 13   ],
                        ])
    var = np.var(data, axis=0)
    print("var",var)
    mult = data.shape[0]*var
    print("mult",mult)
    # sums = np.sum()

if arg == 'for':
    for x in range(3):
        print(x)
    else:
        print('Final x = %d' % (x))
    
    data = np.array([[-4.0, 3.4],
                     [ 3.0, 5.2],
                     [ 1.4, 5.0],
                     [ 3.2, 6.1],
                    ])
    for index, item in enumerate(data):
        print(index, item)

if arg == 'wcss':
    data = np.array([[-4.0, 3.4],
                     [ 3.0, 5.2],
                     [ 1.4, 5.0],
                     [ 3.2, 6.1],
                    ])
    # for i, x in enumerate(data):
    norm = np.linalg.norm(data-np.mean(data, axis=1))
    print(norm)