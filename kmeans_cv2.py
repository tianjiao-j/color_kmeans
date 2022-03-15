import numpy as np
import cv2
from matplotlib import pyplot as plt
import webcolors
from datetime import datetime

img = cv2.imread("images/trousers-black.png")


def detect_colors(image, num_clusters, num_iters, resize_factor, crop_factor, type="hue"):
    height, width, depth = image.shape
    # crop
    crop_factor = (100 - crop_factor) / 2
    image = image[int(height * crop_factor / 100):(height - int(height * crop_factor / 100)),
            int(width * crop_factor / 100):(width - int(width * crop_factor / 100))]
    cv2.imwrite('output/cropped.jpg', image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    start_time = datetime.now()

    w_resize = int(image.shape[1] * resize_factor / 100)  # vertical
    h_resize = int(image.shape[0] * resize_factor / 100)  # horizontal
    dim = (w_resize, h_resize)
    # resize image
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    w, h, d = image.shape
    image_flat = np.reshape(image, (w * h, 3))

    Z = image_flat
    Z = np.float32(Z)

    if (type == "hue"):
        # convert to HSV and use Hue value only
        # resize image
        image_hsv = cv2.resize(image_hsv, dim, interpolation=cv2.INTER_AREA)
        w, h, d = image_hsv.shape
        image_flat_hsv = np.reshape(image_hsv, (w * h, 3))

        Z_hsv = image_flat_hsv
        Z_hsv = np.float32(Z_hsv)
        Z_hsv = Z_hsv[:, 0]

        # define criteria and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, num_iters, 1.0)
        ret, label, center = cv2.kmeans(Z_hsv, num_clusters, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)

        # find centers by RGB values of cluster samples
        for i in range(num_clusters):
            samples = Z[label.ravel() == i]  # RGB values
            center = np.mean(samples, axis=0)
            color = webcolors.rgb_to_hex(center.astype(int))

            # mapping the clusters
            for j in range(len(Z)):
                if (label[j] == i):
                    Z[j] = center

            percentage = len(samples) * 100 / Z.shape[0]
            percentage_str = str(i) + "-" + str(percentage) + "%"
            print "########## cluster info", percentage_str
            print center
            # print(webcolors.hex_to_name(color))  # fixme: color name not found
            plt.scatter(samples[:, 0], samples[:, 1], c=color, label=percentage_str, s=200)
            # plt.scatter(center[0], center[1], s=3000, c='black', marker=r"$ {} $".format(percentage_str))
            plt.legend(loc=2, prop={'size': 20})

        # print("centers", center)
        delay = datetime.now() - start_time
        print "**************** delay", delay.total_seconds()

        # reconstruct image
        Z = Z.reshape((w, h, 3))
        cv2.imwrite('output/mapping.jpg', cv2.cvtColor(Z, cv2.COLOR_RGB2BGR))
        plt.savefig('output/clusters.jpg')
        plt.show()

    elif (type == "rgb"):
        # define criteria and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, num_iters, 1.0)
        ret, label, center = cv2.kmeans(Z, num_clusters, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)

        for i in range(num_clusters):
            color = webcolors.rgb_to_hex(center[i].astype(int))
            samples = Z[label.ravel() == i]

            # mapping the clusters
            for j in range(len(Z)):
                if (label[j] == i):
                    Z[j] = center[i]

            percentage = len(samples) * 100 / Z.shape[0]
            percentage_str = str(i) + "-" + str(percentage) + "%"
            print "########## cluster info", percentage_str
            # print(webcolors.hex_to_name(color))  # fixme: color name not found
            plt.scatter(samples[:, 0], samples[:, 1], c=color, label=percentage_str, s=200)
            # plt.scatter(center[i, 0], center[i, 1], s=3000, c='black', marker=r"$ {} $".format(percentage_str))
            plt.legend(loc=2, prop={'size': 20})

        print "@@@@@@@@@ centers"
        print center
        delay = datetime.now() - start_time
        print "**************** delay", delay.total_seconds()

        # reconstruct image
        Z = Z.reshape((w, h, 3))
        cv2.imwrite('output/mapping.jpg', cv2.cvtColor(Z, cv2.COLOR_RGB2BGR))
        plt.savefig('output/clusters.jpg')
        plt.show()

    else:
        print "Error: type not defined. Type must be either hue or rgb."


# def find_color_name(rgb, table):

detect_colors(img, num_clusters=5, num_iters=50, resize_factor=10, crop_factor=100, type="rgb")
