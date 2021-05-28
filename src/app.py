from cropper import Cropper
from detector import Detector
from reader import Reader
from thread_with_return_value import ThreadWithReturnValue
import os
import cv2
import time

def foo(bar):
    print ('hello {0}'.format(bar))
    return "foo"

if __name__ == "__main__":
    Cropper = Cropper()
    Detector = Detector()
    Reader = Reader()
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
            aligned_image = Cropper.crop_and_align_image(image)
            if aligned_image is False:
                continue
            dictInformationImage = Detector.detect_information(aligned_image)
            dictInformationText = dict()
            dictThreading = dict()
            for key in dictInformationImage.keys():
                # t = time.time()
                
                thread = ThreadWithReturnValue(target=Reader.read_information, args=(dictInformationImage[key],))
                thread.start()
                dictThreading[key] = thread
            
            for key in dictInformationImage.keys():
                # t = time.time()
                dictInformationText[key] = dictThreading[key].join()
            print("Time processing: ", time.time() - start)

            for key in dictInformationText.keys():
                print("{}: {}".format(key, dictInformationText[key]))
        cv2.waitKey()
