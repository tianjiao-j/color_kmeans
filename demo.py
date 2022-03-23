# import the opencv library
import cv2
from kmeans_cv2 import detect_colors
from datetime import datetime
import os
import numpy as np


def make_demo(resize_factor):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

    img = cv2.imread('images/top-white.jpg')

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out1 = cv2.VideoWriter('output/mapping.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                           (int(frame_width * resize_factor), int(frame_height * resize_factor)))
    out2 = cv2.VideoWriter('output/clusters.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                           (frame_width, frame_height))
    # out3 = cv2.VideoWriter('output/combined.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
    # (frame_width, frame_height))

    while (True):
        time = datetime.now()
        time_str = time.strftime("%H:%M:%S")
        date_time = time.strftime("%m_%d_%Y_%H_%M_%S")

        ret, frame = cap.read()
        if ret == True:
            # Write the frame into the file 'output.avi'
            colors_rgb, percentages, mapping, plot = detect_colors(frame, num_clusters=5, num_iters=50,
                                                                   resize_factor=resize_factor * 100,
                                                                   crop_factor=100, type="rgb")
            print('raw ', mapping.shape, plot.shape)
            mappingImages = [mapping, plot, img]
            max_height = 0
            for i in range(len(mappingImages)):
                mapping = mappingImages[i]
                if (mapping.shape[0] > max_height):
                    max_height = mapping.shape[0]
            max_height += 10
            for i in range(len(mappingImages)):
                print(i)
                mapping = mappingImages[i]
                top = int((max_height - mapping.shape[0]) / 2)
                bottom = max_height - mapping.shape[0] - top
                if (top < 0): top = 0
                if (bottom < 0): bottom = 0
                print('top, bottom ', top, bottom)
                mapping = cv2.copyMakeBorder(mapping, top, bottom, 5, 5, cv2.BORDER_CONSTANT)
                if (i == 0):
                    mapping_concat = mapping
                else:
                    print('padded ', mapping_concat.shape, mapping.shape)
                    mapping_concat = np.concatenate((mapping_concat, mapping), axis=1)

            # print mapping.shape, plot.shape
            top = int((plot.shape[0] - mapping.shape[0]) / 2)
            bottom = plot.shape[0] - mapping.shape[0] - top
            left = int((plot.shape[1] - mapping.shape[1]) / 2)
            right = plot.shape[1] - mapping.shape[1] - left
            # mapping = cv2.resize(mapping, (plot.shape[1], plot.shape[0]), interpolation=cv2.INTER_AREA)
            #mapping = cv2.copyMakeBorder(mapping, top, bottom, left, right, cv2.BORDER_CONSTANT)
            # out1.write(mapping)
            # out2.write(plot)
            # Display the resulting frame

            cv2.imshow('mapping_rgb', mapping)
            cv2.imshow('clusters', plot)
            cv2.imshow('concat', mapping_concat)
            if (int(date_time.split("_")[5]) % 10 == 0):
                if (not os.path.exists('test/')):
                    os.mkdir('test/')
                cv2.imwrite('test/mapping_rgb_' + date_time + '.jpg', mapping)
                print(date_time)

            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and video write objects
    cap.release()
    out1.release()
    out2.release()
    # Closes all the frames
    cv2.destroyAllWindows()


make_demo(0.3)
