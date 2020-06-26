import cv2, dlib, argparse
from utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align faces in image')
    parser.add_argument('output', type=str, help='')
    parser.add_argument('--scale', metavar='S', type=float, default=2, help='an integer for the accumulator')
    args = parser.parse_args()

    output_image = args.output
    scale = args.scale

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../../models/shape_predictor_68_face_landmarks.dat")
    video_name = 'test.mp4'
    cap = cv2.VideoCapture(video_name)
    count = 0
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        img = frame
        height, width = img.shape[:2]
        s_height, s_width = height // scale, width // scale
        img = cv2.resize(img, (s_width, s_height))

        dets = detector(img, 1)
        print(dets)

        num_faces = len(dets)
        if num_faces == 0:
            print("Sorry, there were no faces found in '{}'".format(video_name))
            exit()
    
        for i, det in enumerate(dets):
            shape = predictor(img, det)
            left_eye = extract_left_eye_center(shape)
            right_eye = extract_right_eye_center(shape)

            M = get_rotation_matrix(left_eye, right_eye)
            # affine transformation is applied
            rotated = cv2.warpAffine(img, M, (s_width, s_height), flags=cv2.INTER_CUBIC)

            cropped = crop_image(rotated, det)
            number = i + count
            if output_image.endswith('.jpg'):
                output_image_path = output_image.replace('.jpg', '_%i.jpg' % number)
            elif output_image.endswith('.png'):
                output_image_path = output_image.replace('.png', '_%i.jpg' % number)
            else:
                output_image_path = output_image + ('_%i.jpg' % i)
            cv2.imwrite(output_image_path, cropped)
            count = count + 1

cap.release()
cv2.destroyAllWindows()