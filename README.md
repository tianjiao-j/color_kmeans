# color_kmeans

### Usage  
To test the funtion, run  
```
python kmean_cv2.py
```
using kmeans method in cv2  


To use the function in your code:  
```python
from kmeans_cv2 import detect_colors
image = cv2.imread(path_to_image)
## params:
# num_iters: max. number of iterations allowed for kmeans clustering
# resize_factor: 100 = full resolution (not recommended, will be slow)
# crop_factor: 100 = full image
# type: default - clustering based on hue value of pixels, type="rgb" - clustering based on RGB values
detect_colors(image, num_clusters, num_iters, resize_factor, crop_factor, type="hue")
```
