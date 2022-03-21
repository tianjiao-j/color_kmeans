import cv2
import numpy as np

for num_clusters in [0, 3, 5, 10, 20]:
    basename = 'mapping_rgb_'
    # img = np.zeros([1])
    # new_img = np.zeros([1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    if (num_clusters == 0):
        filename = 'images/person1.jpg'
        img = cv2.imread(filename)
        h, w = img.shape[:2]
        print(img.shape)
        cv2.putText(img, 'original image', (10, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('original', img)
        cv2.waitKey(0)
    else:
        new_image = np.zeros((h, w, 3), np.uint8)
        filename = 'output/' + basename + str(num_clusters) + '.jpg'
        new_img = cv2.imread(filename)
        print(new_img.shape)
        title = 'k = ' + str(num_clusters)
        cv2.putText(new_img, title, (10, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow(title, img)
        cv2.waitKey(0)
        img = np.vstack((img, new_img))
        # img = np.concatenate((img, new_img), axis=0)
        cv2.imshow('concat', img)
        cv2.waitKey(0)
        cv2.imwrite('output/concat.jpg', img)
