# import the opencv library
import cv2
from kmeans_cv2 import detect_colors
import numpy as np


def make_demo(resize_factor):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

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

    while (True):
        ret, frame = cap.read()
        if ret == True:
            # Write the frame into the file 'output.avi'
            colors_rgb, percentages, mapping, plot = detect_colors(frame, num_clusters=5, num_iters=50,
                                                                   resize_factor=resize_factor * 100,
                                                                   crop_factor=100, type="rgb")
            # Z = cv2.cvtColor(Z, cv2.COLOR_RGB2BGR)
            # Z = Z.astype(np.uint8)
            print mapping.shape
            out1.write(mapping)
            out2.write(plot)
            # Display the resulting frame
            cv2.imshow('mapping_rgb', mapping)
            cv2.imshow('clusters', plot)

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
