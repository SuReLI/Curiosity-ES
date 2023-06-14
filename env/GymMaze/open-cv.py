import numpy as np, cv2, time 

float_img = np.ones((100,100))
im = np.array(float_img * 255, dtype = np.uint8)

# grayImage = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

# # Start coordinate, here (100, 50)
# # represents the top left corner of rectangle
start_point = (5, 5)
   
# # Ending coordinate, here (125, 80)
# # represents the bottom right corner of rectangle
end_point = (6, 6)
   
# # Black color in BGR
color = (0, 0, 0)
   
# # Line thickness of -1 px
# # Thickness of -1 will fill the entire shape
thickness = -1
   
# # Using cv2.rectangle() method
# # Draw a rectangle of black color of thickness -1 px
image = cv2.rectangle(im, start_point, end_point, color, thickness)

# Center coordinates
center_coordinates = (2, 2)
 
# Radius of circle
radius = 20
  
# Red color in BGR
color = (0, 0, 0)
  
# Line thickness of -1 px
thickness = -1
image= cv2.circle(image, center_coordinates, radius, color, thickness)

print(image)

# # Displaying the image 
cv2.imshow("image", im) 

# # waits for user to press any key
# # (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# # closing all open windows
# cv2.destroyAllWindows()