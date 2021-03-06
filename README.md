# color_kmeans

### Usage  
**To test the funtion, run**  
```
python kmeans_cv2.py
```
it uses kmeans method in cv2  


**To use the function in your code:**    
```python
from kmeans_cv2 import detect_colors
image = cv2.imread(path_to_image)

## params:
# num_iters: max. number of iterations allowed for kmeans clustering
# resize_factor: 100 = full resolution (not recommended, will be slow)
# crop_factor: 100 = full image
# type: default - clustering based on hue value of pixels, type="rgb" - clustering based on RGB values

## returns:
# RGB values of main cluster center, percentage of the main cluster

## other outputs:
# prints color name of main cluster
# prints delay in seconds
# output transformed image after pixel mapping
# output scatterplot of clustered pixels
detect_colors(image, num_clusters, num_iters, resize_factor, crop_factor, type="hue")
```

**Note: it requires the package webcolors and matplotlib which are not avaiable in pepper for visualization purposes, but these packages are not required for the core function.**  

**Reference:**  
https://github.com/jacques-saraydaryan/ros_color_detection  
https://github.com/RoboBreizh-RoboCup-Home/perception-pepper/tree/main/age_color_gender_face_detection
