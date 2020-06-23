import argparse
from deepface import DeepFace


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Detecting landmarks')
    parser.add_argument('--image', dest='image_path', help='Path of image')
    args = parser.parse_args()
    return args


args = parse_args()
image_path = args.image_path
demography = DeepFace.analyze(image_path, actions = ['emotion'])
print("Emotion: ", demography["dominant_emotion"])