import face_alignment
import cv2

def detect_landmarks(frame, fa):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get landmarks of the input image
        imagePoints = fa.get_landmarks_from_image(frame)
        return imagePoints