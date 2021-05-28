from vietocr.tool.translate import build_model, translate, translate_beam_search, process_input, predict
from vietocr.tool.utils import download_weights
from vietocr.tool.config import Cfg
import sys
import os
import cv2
import numpy as np
import math
import pandas as pd
import torch
import time
from cropper import Cropper
from detector import Detector
from format_info import format_information
###multi threading
#from threading import Thread

class Reader():
    def __init__(self):
        config = Cfg.load_config_from_file("../models/models_transformer/vgg-transformer.yml", '../models/models_transformer/base.yml')
        #config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = '../models/models_transformer/my_transformerocr.pth'
        #config['weights'] = 'https://drive.google.com/uc?export=download&id=1-olev206xLgXYf7rnwHrcZLxxLg5rs0p'
        config['device'] = 'cpu'
        config['predictor']['beamsearch'] = False
        #config['cnn']['pretrained']=True
        device = config['device']
        model, vocab = build_model(config)
        weights = ''
        
        if config['weights'].startswith('http'):
            weights = download_weights(config['weights'])
        else:
            weights = config['weights']

        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
        self.config = config
        self.model = model
        self.vocab = vocab
    
    def preprocess_input(self, image):
        """
        param: image: ndarray of image
        """
        h, w, _ = image.shape
        new_w, image_height = self.resize(w, h, self.config['dataset']['image_height'], self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])

        img = cv2.resize(image, (new_w, image_height))
        img = np.transpose(img, (2, 0, 1))
        img = img/255
        return img
    
    def resize(self, w, h, expected_height, image_min_width, image_max_width):
        new_w = int(expected_height * float(w) / float(h))
        round_to = 10
        new_w = math.ceil(new_w / round_to) * round_to
        new_w = max(new_w, image_min_width)
        new_w = min(new_w, image_max_width)

        return new_w, expected_height

    def read_information(self, img):
        img = self.preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        img = torch.FloatTensor(img)
        img = img.to(self.config['device'])

        if self.config['predictor']['beamsearch']:
            sent = translate_beam_search(img, self.model)
            s = sent
        else:
            s = translate(img, self.model)[0].tolist()

        s = self.vocab.decode(s)
        return s




if __name__ == "__main__":
    cropper = Cropper()
    detector = Detector()
    reader = Reader()
    type_img = ['jpg', 'png']
    image_folder = 'test_images/'
    images = os.listdir(image_folder)
    for image_file in images:
        if image_file[-3:] in type_img:
            start = time.time()
            path = image_folder + image_file
            image = cv2.imread(path)
            H, W = image.shape[:2]
            image_resized = cv2.resize(image, (416, int(416 * H/W)))
            cv2.imshow("raw_image", image_resized)
            dictInformationText = dict()
            dictInformationImage = dict()
            return_code, aligned_image = cropper.crop_and_align_image(image)
            #print('cropper: ', time.time() - start)
            tmp = 0
            if return_code == 0:
                for c in detector.classes:
                    dictInformationText[c] = 'N/A'
                print (dictInformationText)
                
            elif return_code == 2:
                tmp = 1
                index = 0
                aligned_image = image
                while(index < 4):
                    dictInformationImage = detector.detect_information(aligned_image)
                    keys = dictInformationImage.keys()
                    if 'id' in keys and 'ho_ten' in keys and 'ngay_sinh' in keys:
                        tmp = 2
                        break
                    else:
                        aligned_image = cv2.rotate(aligned_image, cv2.cv2.ROTATE_90_CLOCKWISE) 
                        index+=1
                
            if tmp == 0:
                dictInformationImage = detector.detect_information(aligned_image)
            if tmp == 1:
                for c in detector.classes:
                    dictInformationText[c] = 'N/A'
                print(dictInformationText)
            #print('detector: ', time.time() - start)
            for key in dictInformationImage.keys():
                dictInformationText[key] = reader.read_information(dictInformationImage[key])
            #cv2.imwrite('images_uploaded/' + dictInformationText['id'] + '.jpg', image)
            output_dict = format_information(dictInformationText)
            print('Time processing: ', time.time() - start)
            for key in output_dict.keys():
                info = key + ': ' + output_dict[key]
                print(info)
            #print(output_dict)
        cv2.waitKey()
    cv2.destroyAllWindows()