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

import lib.FacialExpressionRecognition.transforms as transforms

import sys
sys.path.append('../../')
from models.FacialExpressionRecognition import *

def get_expression(frame):
    cut_size = 44

    transform_test = transforms.Compose([
        transforms.TenCrop(cut_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48,48)).astype(np.uint8)

        img = gray[:, :, np.newaxis]

        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        inputs = transform_test(img)

        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        net = VGG('VGG19')
        checkpoint = torch.load(os.path.join('../models/FacialExpressionRecognition/FER2013_VGG19', 'PrivateTest_model.t7'))
        net.load_state_dict(checkpoint['net'])
        if torch.cuda.is_available():
            net.cuda()
        net.eval()

        ncrops, c, h, w = np.shape(inputs)

        inputs = inputs.view(-1, c, h, w)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        inputs = Variable(inputs)
        outputs = net(inputs)

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg,dim=0)
        _, predicted = torch.max(outputs_avg.data, 0)
        print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))
        expr = [score[0].item(),score[1].item(),score[2].item(),score[3].item(),score[4].item(),score[5].item(),score[6].item()]
        return expr
