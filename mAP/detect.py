import os
import cv2
import numpy as np
import time
import random
import sys


class Detector():
    def __init__(self):
        model_dir = '../../models/models_detect_info/'
        model_weights = model_dir + 'yolov4_tiny_detect_info.weights'
        model_cfg = model_dir + 'yolov4_tiny_detect_info.cfg'
        model_classes = model_dir + 'yolov4_tiny_detect_info.txt'
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.classes = None
        with open(model_classes, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
    def get_output_layers(self):
        layer_names = self.net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        return output_layers

    def crop_image(self, image, x_min, y_min, x_max, y_max):
        cropped_image = image[y_min:y_max, x_min: x_max]
        cv2.rectangle(self.image_original, (x_min, y_min), (x_max, y_max), (0,0,255), 5)
        return cropped_image

    def detect(self, image):
        Width = image.shape[1]
        Height = image.shape[0]


        scale = 0.00392
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers())

        class_ids = []
        confidences = []
        bboxes_pred = []
        bboxes_output = []
        conf_threshold = 0.5
        nms_threshold = 0.5

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    bboxes_pred.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(bboxes_pred, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = bboxes_pred[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            score = confidences[i]
            label = class_ids[i]
            bboxes_output.append([x,y,w,h,score,label])
        return bboxes_output

