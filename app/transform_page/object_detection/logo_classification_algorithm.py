import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
from PIL import Image
import cv2
import torch.nn as nn


class LogoClassifier:
    def __init__(self):
        pass

    def get_model(self, path, n_classes=47):
        '''
            Create a pre-trained ResNet18 model with the fc changed to fit our goal;
        :path: string
            The path to the location on the hard drive where the model is stored;
        :n_classes: int
            The number of classes that the model will classify. The default value is 47;
        '''

        model = torchvision.models.resnet101(pretrained=True)
        in_features = 2048
        model.fc = nn.Linear(in_features, out_features=n_classes)
        model.load_state_dict(torch.load(path))

        return model

    def classify_logos(self, boxes_path, model_path, video_path):
        '''
            Goes through the process of classifying all the logos present in a video;
        :boxes path: string
            The path to the csv containing the coordonates of all the boxes in a video;
        :model_path: string
            The path to the classification model;
        :video_path:
            The path to the video containing logos;
        '''
        #         im_list = []
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        classes = ['ASUS', 'Adidas SB', 'Armani', 'Asics', 'BEATS BY DRE',
                   'BMW', 'BOSCH', 'BOSS', 'Balenciaga', 'BenQ', 'Bentley',
                   'Calvin Klein', 'Chevrolet', 'Converse', 'Corona', 'Dacia',
                   'Everlast', 'Guess', 'Hot Wheels', 'Hugo Boss', 'Hyundai',
                   'IBM', 'LAVAZZA', 'Lamborghini', 'Logitech', "McDonald's",
                   'coca cola', 'iPhone', 'kia da', 'lacoste', 'lego mindstorms',
                   'lexus', 'louis vuitton', 'nestle', 'new balance', 'nivea',
                   'opel', 'pepsi', 'polo ralph lauren', 'prada', 'the north face',
                   'tommy hilfiger', 'toyota', 'uber', 'versace', 'xiaomi', 'zara']

        logo_freq = {}

        model = self.get_model(model_path).to(device=device)
        model.eval()

        boxes = pd.read_csv(boxes_path)

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        frame_cnt = 0
        input_video = cv2.VideoCapture(video_path)
        if input_video.isOpened() == False:
            print("Lol video not found")
        else:
            while (input_video.isOpened()):
                ret, frame = input_video.read()
                if ret == True:
                    boxes_in_frame = boxes[boxes.iloc[:, 0] == frame_cnt]
                    for i in range(len(boxes_in_frame)):
                        x_min, y_min, x_max, y_max = boxes_in_frame.iloc[i, 1:]
                        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                        logo = frame[y_min: y_max, x_min: x_max]
                        logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
                        logo = Image.fromarray(logo)
                        logo = transform(logo)

                        with torch.no_grad():
                            output = model(logo.unsqueeze(0).to(device=device))
                            output = torch.argmax(output)
                            output = classes[int(output)]
                            if output in logo_freq.keys():
                                logo_freq[output] += 1
                            else:
                                logo_freq[output] = 1

                    frame_cnt += 1
                #                     if frame_cnt % 100 == 0:
                #                         print("Frame {}".format(frame_cnt))

                else:
                    break

        logo_freq = dict(sorted(logo_freq.items(), key=lambda x: x[1], reverse=True))
        return logo_freq
