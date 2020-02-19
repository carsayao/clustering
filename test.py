
import os
import sys
from pathlib import Path
sys.path.append("../")


import random
import matplotlib.pyplot as plt
import numpy as np
from read import Read
from kmeans import Kmean
from gmm import GMM

# try:
#     arg = sys.argv[2]
# except IndexError:
#     print(f"You need to type '--test' then arg")
#     sys.exit()
arg = 'gmm'


if arg == 'gmm':
    from scipy.stats import multivariate_normal
    
    # append matrix of matrices
    X = np.array([[1,2],[3,4],[5,6],[7,8]])
    u = np.array([[3,4],[4,5]])
    c = np.array([[.3,.4],[.5,.6]])
    sub = X-u[0]
    dot = np.dot((X-u[0]),np.linalg.inv(c))
    # import pdb; pdb.set_trace()

    # for i in range(1,4):
    #     cov = np.array([[i,i],[i,i]])
    #     a.append(b)
    # print(a)

    gmm = GMM("./data/K-9_means.txt", "./data/K-9_cov.dat")
    # gmm.gauss(prior, mean, cov, x, dim)
    prior = .5
    x = np.array([3,1])
    mean = np.array([4,2])
    cov = np.array([[.5,.4],[-.1,.1]])
    # cov**-1
    # array([[ 1.11111111, -4.44444444],
    #        [ 1.11111111,  5.55555556]])
    # got = gmm.gauss(mean, cov, x)
    gmm.gmm()
    # should = multivariate_normal.pdf(x=mean, mean=mean, cov=cov)
    # print(f"got\n{got}")
    # print(f"should\n{should}")


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

if arg == 'rand':
    grades = np.random.rand((self.SAMPLES, self.K))
    print(grades)

if arg == 'weight':
    x = np.array([.4, .5])
    centroids = np.array()
    norm = np.linalg.norm(x-self.centroids, axis=1)

if arg == 'axis':
    a = np.array([0,1,2,3,4,5,6,7,8,9])
    print("a")
    print(a.shape)
    print(a)
    b = a[:,np.newaxis]
    print("b")
    print(b.shape)
    print(b)
    c = np.arange(20).reshape(10,2)
    print("c")
    print(c.shape)
    print(c)
    d = b * c
    print("d")
    print(d.shape)
    print(d)

if arg == 'paths':
    path = "./test_dir4/fire/mountain/"
    try:
        Path(path).mkdir()
        print(f"[+] {path} created.")
    except FileNotFoundError:
        print(f"[!] {path}'s parent directory does not exist! Defaulting to ./data/.'")
        path = "./data/"
        try:
            Path(path).mkdir(parents=True)
            print(f"[+] {path} created.")
        except FileExistsError:
            print(f"[!] {path} already exists!")
    except FileExistsError:
        print(f"[+] {path} already exists.")
    
if arg == 'array':
    # clusters = [np.zeros(2)] * 5 
    # clusters = np.zeros((5,2))

    print(np.random.randint(5, size=5))
    print(np.random.choice(5, 5, replace=False))

if arg == 'tuple':
    listo = []
    tup = ("hello", .3)
    listo.append(tup)
    tu1 = ("goodbye", .5)
    listo.append(tu1)
    tu2 = ("welcome", .2)
    listo.append(tu2)
    tu3 = ("to the", .4)
    listo.append(tu3)
    import pdb; pdb.set_trace()
    print(f"let me access the array's tuple 'hello'")
    print(listo[0][0])



sys.exit()