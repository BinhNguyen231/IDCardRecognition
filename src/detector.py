import os
import cv2
import numpy as np
import time
import random
import sys

from cropper import Cropper

class Detector():
    def __init__(self):
        model_dir = '../models/models_detect_info/'
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
        #cv2.rectangle(self.image_original, (x_min, y_min), (x_max, y_max), (0,0,255), 5)
        return cropped_image

    def detect_information(self, image):
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

        D = dict()
        image_copy = image.copy()
        for idx, bbox in enumerate(bboxes_output):
            [x,y,w,h,score,label] = bbox
            x1 = max(int(x), 0)
            y1 = max(int(y), 0)
            x2 = min(int(x + w), Width - 1)
            y2 = min(int(y + h), Height - 1)
            cropped_image = image[y1:y2, x1: x2]
            D[self.classes[label]] = cropped_image
            blue = random.randint(0,255)
            green = random.randint(0,255)
            red = random.randint(0,255)
            
            cv2.rectangle(image_copy, (x1,y1), (x2, y2),(blue, green, red) , 2)
            cv2.putText(image_copy, self.classes[label], (x1-3, y1 - 3), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,(blue, green, red) , 1, cv2.LINE_AA)
        cv2.imshow("information_detected", image_copy)
        cv2.imwrite('result/detector_result.jpg', image_copy)
        
        return D

if __name__ == "__main__":
    type_img = ['jpg', 'png']
    image_folder = 'test_images/'
    images = os.listdir(image_folder)
    cropper = Cropper()
    detector = Detector()
    for image_file in images:
        if image_file[-3:] in type_img:
            start = time.time()
            path = image_folder + image_file
            image = cv2.imread(path)
            H, W = image.shape[:2]
            image_resized = cv2.resize(image, (416, int(416 * H/W)))
            cv2.imshow("raw_image", image_resized)
            return_code, aligned_image = cropper.crop_and_align_image(image)
            if return_code != 1:
                continue
            D = detector.detect_information(aligned_image)
            print("Time processing: ", time.time() - start)
            for k in D.keys():
                cv2.imshow(k, D[k])
            #cv2.imwrite(image_folder + 'detector.jpg', aligned_image)
            cv2.waitKey()
    cv2.destroyAllWindows()
    