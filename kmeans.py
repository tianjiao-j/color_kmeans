import numpy as np
import numpy.linalg as alg
import cv2
from Cluster import Cluster
import random as rand
from datetime import datetime
import webcolors
from matplotlib import pyplot as plt

start_time = datetime.now()
image = cv2.imread("images/colors.jpeg")
# print(image.shape)
w_resize = 100
h_resize = 100
image = cv2.resize(image, (w_resize, h_resize), interpolation=cv2.INTER_AREA)
w, h, d = image.shape
image_flat = np.reshape(image, (w * h, 3))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#image_flat = image_flat / 255.0


# X: Nxd
def kmeans(X, num_clusters):
    N, d = X.shape
    # initialize the initial cluster centers
    centers = []
    for _ in range(num_clusters):
        centers.append([0, 0, 0])
        # centers.append(np.ndarray.tolist(X[rand.randint(0, 9999)]))

    # perform k-means
    clusters = []
    for _ in range(num_clusters):
        cluster = Cluster([])
        clusters.append(cluster)

    loss = 0
    count = 0

    while True:
        for cluster in clusters:
            cluster.samples = []

        new_loss = 0

        for x in X:
            dists = []
            for center in centers:
                np_center = np.array(center)
                new_loss += alg.norm(np_center - x)
                dists.append(alg.norm(np_center - x))
            min_dist = np.min(dists)
            min_idx = 0
            for idx in range(num_clusters):
                if (min_dist == dists[idx]):
                    min_idx = idx
                    break
            clusters[min_idx].samples.append(np.ndarray.tolist(x))

        for _ in range(num_clusters):
            clusters[_].samples.append(centers[_])

        # calculate new centers
        new_centers = []
        for cluster in clusters:
            samples = np.array(cluster.samples)
            print(len(samples))
            new_centers.append(np.mean(samples, axis=0))

        # print(centers)
        # print(new_centers)
        print(loss)
        print(new_loss)

        if loss == new_loss or count == 1:
            print("kmeans complete")
            break
        else:
            centers = new_centers
            loss = new_loss

    return centers, clusters


num_clusters = 5
centers, clusters = kmeans(image_flat, num_clusters)
#for i in range(len(centers)):
    #centers[i] *= 255
print(centers)  # cluster centers in RGB values
print(datetime.now() - start_time)  # delay

for i in range(num_clusters):
    color = webcolors.rgb_to_hex(centers[i].astype(int))
    samples = np.array(clusters[i].samples)
    #samples = samples * 255
    center = np.array(centers[i])
    #print(samples[:, 1])
    plt.scatter(samples[:, 0], samples[:, 1], c=color)
    plt.scatter(center[0], center[1], s=80, c='y', marker='s')

plt.show()
