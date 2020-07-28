import argparse
import numpy as np
import cv2
import face_alignment
import pathlib

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Detecting landmarks')
    parser.add_argument('--video', dest='video_path', help='Path of the video')
    parser.add_argument('--subdirectory', dest='subdirectory', help='Path of the subdirectory')
    parser.add_argument('--n_seconds', dest='n_seconds', help='Extract frames every n seconds')
    args = parser.parse_args()
    return args

def main():
    # Initialize the chip resolution
    chipSize = 512
    chipCorners = np.float32([[0,0],
                            [chipSize,0],
                            [0,chipSize],
                            [chipSize,chipSize]])

    # Initialize the face alignment tracker
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=True,  device='cpu')
    count = 0
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    seconds = n_seconds
    fps = cap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
    multiplier = fps * seconds

    while ret:
        frameId = int(round(cap.get(1)))
        ret, frame = cap.read()
        count = count +1
        # Run the face alignment tracker
        if(ret):
            imagePoints = fa.get_landmarks_from_image(frame)
            if(imagePoints is not None):
                imagePoints = imagePoints[0]

                # Compute the Anchor Landmarks
                # This ensures the eyes and chin will not move within the chip
                rightEyeMean = np.mean(imagePoints[36:42], axis=0)
                leftEyeMean  = np.mean(imagePoints[42:47], axis=0)
                middleEye    = (rightEyeMean + leftEyeMean) * 0.5
                chin         = imagePoints[8]

                # Compute the chip center and up/side vectors
                mean = ((middleEye * 3) + chin) * 0.25
                centered = imagePoints - mean 
                rightVector = (leftEyeMean - rightEyeMean)
                upVector    = (chin        - middleEye)

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
        
            if frameId % multiplier == 0:
                print("Saving face... ")
                print(count)
                path = pathlib.Path('extracted_faces/'+subdirectory).mkdir(parents=True, exist_ok=True) 
                imageName = path + "frame%d.jpg" % frameId
                cv2.imwrite(imageName, chip)
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    video_path = args.video_path
    subdirectory = args.subdirectory
    n_seconds = int(args.n_seconds)
    main()