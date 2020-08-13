import cv2
import numpy as np

def create_mask(image):
    # Create mask and draw circle onto mask
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

    # Crop mask and turn background
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    x,y,w,h = cv2.boundingRect(mask)
    result = ROI[y:y+h,x:x+w]
    mask = mask[y:y+h,x:x+w]
    result[mask==0] = (0,0,0)

    return result