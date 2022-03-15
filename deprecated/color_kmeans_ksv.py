print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
import cv2
from skimage import io, color
from skimage import img_as_float
from skimage.viewer import ImageViewer


class ClusterLocal:
    def __init__(self, label):
        self.label = label
        self.centroid_value = []
        self.nbPixel = 0
        self.pixelList = []
        self.S_sum = 0
        self.V_sum = 0
        self.max_V_value = 0
        self.max_V_pixel = (0, 0, 0)

    def updateSAvg(self, s):
        self.S_sum = self.S_sum + s

    def updateMaxVPixel(self, V, pixel):
        if self.max_V_value < V:
            self.max_V_value = V
            self.max_V_pixel = pixel

    def updateVAvg(self, v):
        self.V_sum = self.V_sum + v

    def getSAvg(self):
        if len(self.pixelList) == 0:
            return 0
        return float(self.S_sum) / float(len(self.pixelList))

    def getVAvg(self):
        if len(self.pixelList) == 0:
            return 0
        return float(self.V_sum) / float(len(self.pixelList))

    def getValue(self):
        (H, S, V) = self.centroid_value
        S = self.getSAvg()
        V = self.getVAvg()
        return (H, S, V)

    def getNbPixel(self):
        return len(self.pixelList)


class KmeansColorDetection():
    _n_color = 3
    _isImgDisplayed = False

    def __init__(self, n_color, isImgDisplayed):
        self._n_color = n_color
        self._isImgDisplayed = isImgDisplayed
        self.fig = None

    def process_image(self, img):
        imgRGB = img[:, :, ::-1]

        # convert to hsv to cluster only on H
        imgHSV = color.rgb2hsv(imgRGB)

        # Load Image and transform to a 2D numpy array.
        w, h, d = original_shape = tuple(imgHSV.shape)

        # Convert to floats instead of the default 8 bits integer coding. Dividing by
        # 255 is important so that plt.imshow behaves works well on float data (need to
        # be in the range [0-1]

        imgHSV = np.array(imgHSV, dtype=np.float64)
        imgHSVOnlyH = np.copy(imgHSV)

        # Fix the S and V value of HSV
        imgHSVOnlyH[:, :, 0] = imgHSV[:, :, 0]
        imgHSVOnlyH[:, :, 1] = 0.5
        imgHSVOnlyH[:, :, 2] = 0.5

        assert d == 3
        image_array = np.reshape(imgHSVOnlyH, (w * h, d))

        # print("Fitting model on a small sub-sample of the data")
        t0 = time()
        image_array_sample = shuffle(image_array, random_state=0)[:1000]
        kmeans = KMeans(n_clusters=self._n_color, tol=5, random_state=0).fit(image_array_sample)
        # print("done in %0.3fs." % (time() - t0))

        # Get labels for all points
        # print("Predicting color indices on the full image (k-means)")
        t0 = time()
        labels = kmeans.predict(image_array)
        # print("done in %0.3fs." % (time() - t0))

        clusters = self.process_kmean_result(imgHSV, kmeans.cluster_centers_, labels, w, h)

        if self._isImgDisplayed:
            # curently not working plt.show work only in main thread
            self.displayResult(imgHSV, imgHSVOnlyH, kmeans.cluster_centers_, clusters, w, h)
            pass
        return clusters

    def recreate_image(self, codebook, clusters, w, h, cluster_label):
        """Recreate the (compressed) image from the code book & labels"""
        d = codebook.shape[1]
        image = np.zeros((w, h, d))
        for i in range(w):
            for j in range(h):
                image[i][j] = (0, 0, 1)

        for cluster in clusters.values():
            if (cluster.label == cluster_label or cluster_label == -1):
                for (i, j) in cluster.pixelList:
                    image[i][j] = cluster.getValue()
        return image

    def process_kmean_result(self, originalimg, codebook, labels, w, h):
        """Recreate the (compressed) image from the code book & labels"""
        d = codebook.shape[1]
        image = np.zeros((w, h, d))
        # get the average of the saturation and Value
        # cation use gloal staturation and value on the whole image
        # S_avg=np.average(originalimg[:,:,1])
        # V_avg=np.average(originalimg[:,:,2])
        label_idx = 0
        clusters = {}
        for n in range(len(codebook)):
            cluster = ClusterLocal(n)
            cluster.centroid_value = codebook[n]
            clusters[n] = cluster

        for i in range(w):
            for j in range(h):
                cluster_id = labels[label_idx]
                clusters[cluster_id].pixelList.append((i, j))
                clusters[cluster_id].updateSAvg(originalimg[i, j][1])
                clusters[cluster_id].updateVAvg(originalimg[i, j][2])
                clusters[cluster_id].updateMaxVPixel(originalimg[i, j][2], originalimg[i, j])

                label_idx += 1
        pixelSum = 0
        for cluster in clusters.values():
            pixelSum = pixelSum + cluster.getNbPixel()

        # To demove is to many times
        for cluster in clusters.values():
            data = np.zeros(shape=(1, 1, 3), dtype=np.float64)
            data[0, 0, :] = cluster.getValue()
            # print "Cluster[%s]: HSV:%s, RGB:%s, nbPixel:[%s], imgPercentage:%s" % (str(cluster.label),str((cluster.getValue()[0]*360,cluster.getValue()[1]*100,cluster.getValue()[2]*100)),str(color.hsv2rgb(data)*255),str(cluster.getNbPixel()),str(float(cluster.getNbPixel())/float(pixelSum)))

        return clusters

    def displayResult(self, imgHSV, imgHSVOnlyH, codebook, clusters, w, h):

        # im1 = Image.frombytes("RGB", (w, h), imgHSV)
        # im1.show()

        # viewer = ImageViewer(color.hsv2rgb(imgHSV))
        # viewer.show()
        # cv2.imshow('image',imgHSV)

        if (self.fig == None):
            self.fig = plt.figure(figsize=(8, 2))
        # plt.clf()
        # ax = plt.axes([0, 0, 1, 1])
        plt.axis('off')
        # plt.title('Original image imgHSV')
        a = self.fig.add_subplot(1, 4, 1)
        a.set_title('Original image', fontsize=8)
        plt.imshow(color.hsv2rgb(imgHSV), aspect='auto')
        plt.pause(.1)
        # plt.tight_layout()

        # plt.figure(2)
        # plt.clf()
        # ax = plt.axes([0, 0, 1, 1])
        plt.axis('off')
        # plt.title('Original image imgHSVOnlyH')
        b = self.fig.add_subplot(1, 4, 2)
        b.set_title('Image with only H set (HSV)', fontsize=8)
        plt.imshow(color.hsv2rgb(imgHSVOnlyH), aspect='auto')
        plt.pause(.1)
        # plt.tight_layout()

        # plt.figure(3)
        # plt.clf()
        # ax = plt.axes([0, 0, 1, 1])
        plt.axis('off')
        # plt.title('Quantized image (64 colors, K-Means)')
        image = self.recreate_image(codebook, clusters, w, h, -1)
        max_size = 0
        label_max = 0
        for cluster in clusters.values():
            if cluster.getNbPixel() > max_size:
                max_size = cluster.getNbPixel()
                label_max = cluster.label
        c = self.fig.add_subplot(1, 4, 3)
        c.set_title('K-Means ' + str(self._n_color) + ' clusters', fontsize=8)
        plt.imshow(color.hsv2rgb(image), aspect='auto')
        plt.pause(.1)
        # plt.tight_layout()

        # plt.figure(4)
        # plt.clf()
        # ax = plt.axes([0, 0, 1, 1])
        plt.axis('off')
        # plt.title('Quantized image (64 colors, K-Means)')
        image2 = self.recreate_image(codebook, clusters, w, h, label_max)
        d = self.fig.add_subplot(1, 4, 4)
        d.set_title('Main color cluster', fontsize=8)
        plt.imshow(color.hsv2rgb(image2), aspect='auto')
        plt.pause(.1)
        plt.axis('off')
        # plt.tight_layout()

        # plt.show()