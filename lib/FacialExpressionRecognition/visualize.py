import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
from torch.autograd import Variable

import transforms as transforms

import sys
sys.path.append('../../')
from models.FacialExpressionRecognition import *


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Detecting facial expression')
    parser.add_argument('--image', dest='image_path', help='Path of image')
    parser.add_argument('--output', dest='output_path', help='Path of the output')
    args = parser.parse_args()
    return args


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def draw(raw_img,class_names,score):    
    plt.rcParams['figure.figsize'] = (13.5,5.5)
    axes=plt.subplot(1, 3, 1)
    plt.imshow(raw_img)
    plt.xlabel('Input Image', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()

    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)

    plt.subplot(1, 3, 2)
    ind = 0.1+0.6*np.arange(len(class_names))    # the x locations for the groups
    width = 0.4       # the width of the bars: can also be len(x) sequence
    color_list = ['red','orangered','darkorange','limegreen','darkgreen','royalblue','navy']
    for i in range(len(class_names)):
        plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
    plt.title("Classification results ",fontsize=20)
    plt.xlabel(" Expression Category ",fontsize=16)
    plt.ylabel(" Classification Score ",fontsize=16)
    plt.xticks(ind, class_names, rotation=45, fontsize=14)

    plt.savefig(os.path.join(output_path))
    plt.close()
    
    

def main():
    cut_size = 44

    transform_test = transforms.Compose([
        transforms.TenCrop(cut_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    raw_img = cv2.imread(image_path)[...,::-1]
    gray = rgb2gray(raw_img)
    gray = cv2.resize(gray, (48,48)).astype(np.uint8)

    img = gray[:, :, np.newaxis]

    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    net = VGG('VGG19')
    checkpoint = torch.load(os.path.join('../../models/FacialExpressionRecognition/FER2013_VGG19', 'PrivateTest_model.t7'))
    net.load_state_dict(checkpoint['net'])
    net.cuda()
    net.eval()

    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.cuda()
    inputs = Variable(inputs)
    outputs = net(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

    score = F.softmax(outputs_avg,dim=0)
    _, predicted = torch.max(outputs_avg.data, 0)
    draw(raw_img,class_names,score)
    print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))
    with open("../../src/expression.txt", "ab") as f:
        np.savetxt(f, [[outputs_avg[0].item()],[outputs_avg[1].item()],[outputs_avg[2].item()],[outputs_avg[3].item()],[outputs_avg[4].item()],[outputs_avg[5].item()],[outputs_avg[6].item()]])

if __name__ == '__main__':
    args = parse_args()
    image_path = "../../"+args.image_path
    output_path = args.output_path
    main()
