import os
import cv2
from datetime import datetime
import re
from random import randint


# compare the creation time of 2 images
# params: 2 strings for names of images
def compare_img_datetime(img1, img2):
    img1 = img1.replace('mapping_rgb_', '')
    img1 = img1.replace('.jpg', '')
    img2 = img2.replace('mapping_rgb_', '')
    img2 = img2.replace('.jpg', '')
    datetime1 = datetime.strptime(img1, '%m_%d_%Y_%H_%M_%S')
    datetime2 = datetime.strptime(img2, '%m_%d_%Y_%H_%M_%S')
    if (datetime1 < datetime2):  # img1 is created earlier
        return True
    else:
        return False


output_path = 'test/'
all_files = os.listdir(output_path)
all_images = []
for file in all_files:
    if ('.jpg' in file and 'mapping_rgb' in file):
        #print("image found")
        if (len(all_images) == 0):
            all_images.append(file)
        else:
            inserted = False
            for i in range(len(all_images)):
                if (compare_img_datetime(file, all_images[i])):
                    #print(file + " is earlier than " + all_images[i])
                    all_images.insert(i, file)
                    inserted = True
                    break
            if (not inserted):
                all_images.append(file)
print(all_images)

while True:
    image_path = output_path + all_images[0]
    img = cv2.imread(image_path)
    cv2.imshow('replay', img)
    os.remove(image_path)
    all_images = all_images[1:]
    if (len(all_images) == 0):
        print("no images available")
        break
    # # Press Q on keyboard to stop recording
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    key = cv2.waitKey(5000)  # pauses for 3 seconds before fetching next image
    if key == 27:  # if ESC is pressed, exit loop
        cv2.destroyAllWindows()
        break
