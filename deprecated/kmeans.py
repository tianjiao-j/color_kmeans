import numpy as np
import numpy.linalg as alg
import cv2
from Cluster import Cluster
import random as rand
from datetime import datetime
import webcolors
from matplotlib import pyplot as plt

start_time = datetime.now()
image = cv2.imread("images/top-white.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

scale_percent = 30  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# print(image.shape)
# w_resize = 100
# h_resize = 100
# image = cv2.resize(image, (w_resize, h_resize), interpolation=cv2.INTER_AREA)

w, h, d = image.shape
image_flat = np.reshape(image, (w * h, 3))


# image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# image_flat = image_flat / 255.0


# X: Nxd
def kmeans(X, num_clusters, num_iters=10000):
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

    while count < num_iters:
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
            curr_samples = np.array(cluster.samples)
            print("number of samples in cluster:", len(curr_samples))
            new_centers.append(np.mean(curr_samples, axis=0))

        # print(centers)
        # print(new_centers)
        print("loss", loss)
        print("new_loss", new_loss)

        if loss == new_loss:
            print("kmeans complete")
            print("num_iters", count)
            break
        else:
            centers = new_centers
            loss = new_loss
            count += 1

    return centers, clusters


num_clusters = 3
centers, clusters = kmeans(image_flat, num_clusters, num_iters=5)
# for i in range(len(centers)):
# centers[i] *= 255
print(centers)  # cluster centers in RGB values
delay = datetime.now() - start_time
print("delay", delay.total_seconds())  # delay

for i in range(num_clusters):
    color = webcolors.rgb_to_hex(centers[i].astype(int))
    samples = np.array(clusters[i].samples)
    # samples = samples * 255
    center = np.array(centers[i])
    print(webcolors.rgb_to_hex(center.astype(int)))
    # print(samples[:, 1])
    plt.scatter(samples[:, 0], samples[:, 1], c=color)
    plt.scatter(center[0], center[1], s=80, c='y', marker=r"$ {} $".format(i))

plt.show()
