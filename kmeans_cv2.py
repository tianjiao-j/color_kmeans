import numpy as np
import cv2
from matplotlib import pyplot as plt
import webcolors
from datetime import datetime

img = cv2.imread("images/face.jpeg")


def detect_colors(image, num_clusters, num_iters, resize_factor, crop_factor):
    height, width, depth = image.shape
    # crop
    crop_factor = (100 - crop_factor) / 2
    image = image[int(height * crop_factor / 100):(height - int(height * crop_factor / 100)),
            int(width * crop_factor / 100):(width - int(width * crop_factor / 100))]
    cv2.imwrite('output/cropped.jpg', image)
    # cv2.imshow('output/cropped.jpg', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    start_time = datetime.now()

    # scale_percent = 30  # percent of original size
    w_resize = int(image.shape[1] * resize_factor / 100)  # vertical
    h_resize = int(image.shape[0] * resize_factor / 100)  # horizontal
    dim = (w_resize, h_resize)
    # resize image
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    w, h, d = image.shape
    image_flat = np.reshape(image, (w * h, 3))

    Z = image_flat
    Z = np.float32(Z)
    print(Z.shape)
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
        print("cluster info", percentage_str)
        # print(webcolors.hex_to_name(color))  # fixme: color name not found
        plt.scatter(samples[:, 0], samples[:, 1], c=color)
        plt.scatter(center[i, 0], center[i, 1], s=3000, c='black', marker=r"$ {} $".format(percentage_str))
    print("centers", center)
    delay = datetime.now() - start_time
    print("delay", delay.total_seconds())

    # reconstruct image
    Z = Z.reshape((w, h, 3))
    # print(Z.shape)
    cv2.imwrite('output/mapping.jpg', cv2.cvtColor(Z, cv2.COLOR_RGB2BGR))
    # cv2.imshow("mapping", Z)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.savefig('output/clusters.jpg')
    plt.show()

    # img1 = cv2.imread("output/cropped.jpg")
    # img2 = cv2.imread("output/mapping.jpg")
    # img3 = cv2.imread("output/clusters.jpg")
    #
    # imgh = np.vstack([img1, img2, img3])
    # cv2.imshow("all", imgh)
    # cv2.imwrite("output/all_fig.jpg", imgh)

detect_colors(img, num_clusters=5, num_iters=50, resize_factor=50, crop_factor=50)
