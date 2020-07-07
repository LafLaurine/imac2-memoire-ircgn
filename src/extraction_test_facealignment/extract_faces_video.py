import numpy as np
import cv2
import face_alignment
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Detecting landmarks')
    parser.add_argument('--image', dest='image_path', help='Path of image')
    args = parser.parse_args()
    return args

# Initialize the chip resolution
chipSize = 256
chipCorners = np.float32([[0,0],
                          [chipSize,0],
                          [0,chipSize],
                          [chipSize,chipSize]])

# Initialize the face alignment tracker
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, face_detector='sfd', device="cuda")

cap = cv2.VideoCapture("test.avi")
ret, frame = cap.read()
count = 0
while(ret):
    # Run the face alignment tracker on the frame
    imagePoints = fa.get_landmarks_from_image(frame)
    if(imagePoints is not None):
        imagePoints = imagePoints[0]

        # Compute the Anchor Landmarks
        # This ensures the eyes and chin will not move within the chip
        rightEyeMean = np.mean(imagePoints[36:42], axis=0)
        leftEyeMean  = np.mean(imagePoints[42:47], axis=0)
        middleEye    = (rightEyeMean + leftEyeMean) * 0.5
        chin         = imagePoints[8]
        #cv2.circle(frame, tuple(rightEyeMean[:2].astype(int)), 30, (255,255,0))
        #cv2.circle(frame, tuple(leftEyeMean [:2].astype(int)), 30, (255,0,255))

        # Compute the chip center and up/side vectors
        mean = ((middleEye * 3) + chin) * 0.25
        centered = imagePoints - mean 
        rightVector = (leftEyeMean - rightEyeMean)
        upVector = (chin - middleEye)

        # Divide by the length ratio to ensure a square aspect ratio
        rightVector /= np.linalg.norm(rightVector) / np.linalg.norm(upVector)

        # Compute the corners of the facial chip
        imageCorners = np.float32([(mean + ((-rightVector - upVector)))[:2],
                                    (mean + (( rightVector - upVector)))[:2],
                                    (mean + ((-rightVector + upVector)))[:2],
                                    (mean + (( rightVector + upVector)))[:2]])

        # Compute the Perspective Homography and Extract the chip from the image
        chipMatrix = cv2.getPerspectiveTransform(imageCorners, chipCorners)
        chip = cv2.warpPerspective(frame, chipMatrix, (chipSize, chipSize))

        cv2.imwrite("frame%d.jpg" % count, chip)
        ret,image = cap.read()
        print ('Read a new frame: ', ret)
        count += 2

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
