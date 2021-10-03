# Importing all needed libraries.
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageDraw, ImageFilter
import pandas as pd
import os

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

from .utils.engine import train_one_epoch, evaluate
from .utils import utils
from .utils import transforms as T
from PIL.ExifTags import TAGS, GPSTAGS
import cv2
import re


class ODM:

    def __init__(self):
        pass

    def get_model(self, num_classes, path_to_model):
        '''
            This function is used to create a pretrained FastRCNN Model.
        :num_classes: int
            The number of classes. Should be setted n+1 wher n is the number of object to detect.
        '''
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model.load_state_dict(torch.load(path_to_model))

        return model

    def detect_logos(self, path_to_model, path_to_video,
                     path_to_save, file_name):  # path to save maybe be with os.path.join   # create a csv file with frames and boxes

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded_model = self.get_model(num_classes=2, path_to_model=path_to_model)
        loaded_model.to(device)
        loaded_model.eval()
        cap = cv2.VideoCapture(path_to_video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(os.path.join(path_to_save, re.findall(r'(\w+).mp4', file_name)[0] + '.mp4'), cv2.VideoWriter_fourcc(*"a\0\0\0"), fps,
                              (int(cap.get(3)), int(cap.get(4))))

        transform = T.Compose([T.ToTensor()])

        frame_nr = 0
        boxes_df = pd.DataFrame()
        # when iterate thorugh every box, append row with frame_nr anb box coords
        while cap.isOpened():

            ret, frame = cap.read()

            if ret == False:
                break

            img = Image.fromarray(frame).convert('RGB')

            frame = transform(img, '0')

            img, _ = frame
            # Making the prediction.
            with torch.no_grad():
                prediction = loaded_model([img.to(device)])

            # Getting an drawing the image.
            image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

            for element in range(len(prediction[0]['boxes'])):
                boxes = prediction[0]['boxes'][element].cpu().numpy()
                score = np.round(prediction[0]['scores'][element].cpu().numpy(), decimals=4)

                if score > 0.3:

                    box = (int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3]))

                    new_row = pd.Series([frame_nr, box[0], box[1], box[2], box[3]])
                    boxes_df = boxes_df.append(new_row, ignore_index=True)

                    ic = image.crop(box)
                    for i in range(10):
                        ic = ic.filter(ImageFilter.BLUR)
                    image.paste(ic, box)

                    # image[] = original_image.filter(ImageFilter.BLUR)
                    # blur(image,(int(boxes[0]), int(boxes[2]),int(boxes[1]), int(boxes[3])))

            frame = np.array(image)
            # blur(frame,coords)
            frame_nr += 1

            out.write(frame)

            # cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        boxes_df.to_csv(os.path.join(path_to_save, 'logo_boxes.csv'), index=False)

        out.release()
        cap.release()
        cv2.destroyAllWindows()


# class ODM:
#
#     def __init__(self):
#         pass
#
#     def get_model(self, num_classes, path_to_model):
#         '''
#             This function is used to create a pretrained FastRCNN Model.
#         :num_classes: int
#             The number of classes. Should be setted n+1 wher n is the number of object to detect.
#         '''
#         model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#
#         # get the number of input features for the classifier
#         in_features = model.roi_heads.box_predictor.cls_score.in_features
#
#         # replace the pre-trained head with a new one
#         model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#
#         model.load_state_dict(torch.load(path_to_model))
#
#         return model
#
#     def detect_logos(self, path_to_model, path_to_video, path_to_save):  # path to save maybe be with os.path.join   # create a csv file with frames and boxes
#
#         '''
#             Function to find the logos and apply the transformations.
#         :param path_to_model: str
#             Parameter that represents the path to the trained model.
#         :param path_to_video: str
#             Parameter that represents the path to media file.
#         :param path_to_save: str
#             Parameter that represents the path to the directory where the media file should be saved.
#         :return:
#         '''
#
#         device = torch.device('cuda')
#         loaded_model = self.get_model(num_classes=2, path_to_model=path_to_model)
#         loaded_model.to(device)
#         loaded_model.eval()
#         cap = cv2.VideoCapture(path_to_video)
#         out = cv2.VideoWriter(os.path.join(path_to_save, 'blured.avi'), cv2.VideoWriter_fourcc(*"MJPG"), 30,
#                               (int(cap.get(3)), int(cap.get(4))))
#
#         transform = T.Compose([T.ToTensor()])
#
#         frame_nr = 0
#         boxes_df = pd.DataFrame()
#         # when iterate thorugh every box, append row with frame_nr anb box coords
#         while cap.isOpened():
#             ret, frame = cap.read()
#
#             img = Image.fromarray(frame).convert('RGB')
#
#             frame = transform(img, '0')
#
#             img, _ = frame
#             # Making the prediction.
#             with torch.no_grad():
#                 prediction = loaded_model([img.to(device)])
#
#             # Getting an drawing the image.
#             image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
#
#             for element in range(len(prediction[0]['boxes'])):
#                 boxes = prediction[0]['boxes'][element].cpu().numpy()
#                 score = np.round(prediction[0]['scores'][element].cpu().numpy(), decimals=4)
#
#                 if score > 0.3:
#
#                     box = (int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3]))
#
#                     new_row = pd.Series([frame_nr, box[0], box[1], box[2], box[3]])
#                     boxes_df = boxes_df.append(new_row, ignore_index=True)
#
#                     ic = image.crop(box)
#                     for i in range(
#                             10):  # with the BLUR filter, you can blur a few times to get the effect you're seeking
#                         ic = ic.filter(ImageFilter.BLUR)
#                     image.paste(ic, box)
#
#                     # image[] = original_image.filter(ImageFilter.BLUR)
#                     # blur(image,(int(boxes[0]), int(boxes[2]),int(boxes[1]), int(boxes[3])))
#
#             frame = np.array(image)
#             # blur(frame,coords)
#
#             if ret == True:
#                 out.write(frame)
#             # cv2.imshow('frame',frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#         boxes_df.to_csv(os.path.join(path_to_save, 'logo_boxes.csv'), index=False)
#
#         out.release()
#         cap.release()
#         cv2.destroyAllWindows()
