import cv2
import numpy as np


# Create mask and draw circle onto mask
image = cv2.imread('extraction/extracted_faces/Pierre/000001.jpg')
mask = np.zeros(image.shape, dtype=np.uint8)
width = image.shape[0]
hight = image.shape[1]
center_coordinates = ( int(width/2),int(hight/2)) 
axesLength = (200, 250)
angle = 0
startAngle = 0
endAngle = 360
cv2.ellipse(mask, center_coordinates,axesLength, angle, 
                          startAngle, endAngle, (255,255,255), -1)

# Bitwise-and for ROI
ROI = cv2.bitwise_and(image, mask)

# Crop mask and turn background white
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
x,y,w,h = cv2.boundingRect(mask)
result = ROI[y:y+h,x:x+w]
mask = mask[y:y+h,x:x+w]
result[mask==0] = (255,255,255)

cv2.imwrite('result.jpg', result)
cv2.waitKey()