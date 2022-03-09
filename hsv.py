import cv2

# load the original image and show it
image = cv2.imread('images/colors.jpeg')
cv2.imshow("RGB", image)
# loop over each of the individual channels and display them
for (name, chan) in zip(("B", "G", "R"), cv2.split(image)):
	cv2.imshow(name, chan)
# wait for a keypress, then close all open windows
cv2.waitKey(0)
cv2.destroyAllWindows()
# convert the image to the HSV color space and show it
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)
# loop over each of the individual channels and display them
for (name, chan) in zip(("H", "S", "V"), cv2.split(hsv)):
	cv2.imshow(name, chan)
# wait for a keypress, then close all open windows
cv2.waitKey(0)
cv2.destroyAllWindows()