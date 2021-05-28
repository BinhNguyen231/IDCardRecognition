import os
import cv2
import numpy as np
import time
import math

class Cropper():
    def __init__(self):
        model_dir = '../models/models_detect_4points/'
        model_weights = model_dir + 'yolov4_detect_4points_last.weights'
        model_cfg = model_dir + 'yolov4_detect_4points_last.cfg'
        model_classes = model_dir + 'yolov4_detect_4points_last.txt'
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.classes = None
        with open(model_classes, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        return output_layers
    
    def compute_distance(self, p1, p2):
        return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

    def check_4points_location(self, bbox_list):
        top_left, top_right, bottom_left, bottom_right, quoc_huy = bbox_list
        if self.compute_distance(top_left, quoc_huy) < self.compute_distance(bottom_left, quoc_huy) and \
            self.compute_distance(bottom_left, quoc_huy) < self.compute_distance(top_right, quoc_huy) and \
            self.compute_distance(top_right, quoc_huy) < self.compute_distance(bottom_right, quoc_huy):
            return True
        return False

    def locate_4points(self, bbox_list):
        p1, p2, p3, p4, quoc_huy = bbox_list
        bbox_result = []
        dis_1 = self.compute_distance(p1, quoc_huy)
        dis_2 = self.compute_distance(p2, quoc_huy)
        dis_3 = self.compute_distance(p3, quoc_huy)
        dis_4 = self.compute_distance(p4, quoc_huy)
        distance_list = [dis_1, dis_2, dis_3, dis_4]
        arg_sort = np.argsort(distance_list)
        tl = bbox_list[arg_sort[0]]
        bl = bbox_list[arg_sort[1]]
        tr = bbox_list[arg_sort[2]]
        br = bbox_list[arg_sort[3]]
        bbox_result = [tl, tr, bl, br, quoc_huy]
        return bbox_result

    def compute_angle_from_3_points(self, a, b, c):
        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return ang + 360 if ang < 0 else ang

    def find_min_distance(self, center, points):
        min_distance = 100000
        point_return = None
        for point in points:
            dis = self.compute_distance(center, point)
            if dis < min_distance:
                min_distance = dis
                point_return = point
        return point_return


    def erase_corner_redundant(self, bbox_list):
        quoc_huy = bbox_list[-1]
        distance_list = []
        for i in range(len(bbox_list)-1):
            distance = self.compute_distance(bbox_list[i], quoc_huy)
            distance_list.append(distance)
        idx_top_left = np.argmin(distance_list)
        top_left = bbox_list[idx_top_left]
        bbox_list.pop(-1)
        bbox_list.pop(idx_top_left)
        list_top_right = []
        list_bottom_right = []
        list_bottom_left = []
        for i, box in enumerate(bbox_list):
            angle = self.compute_angle_from_3_points(top_left,quoc_huy, box)
            if angle > 135 and angle <= 165:
                list_top_right.append(box)
            elif (angle > 165 and angle  < 200):
                list_bottom_right.append(box)
            elif (angle > 250 and angle < 280):
                list_bottom_left.append(box)
        top_right = self.find_min_distance(quoc_huy, list_top_right)
        bottom_left = self.find_min_distance(quoc_huy, list_bottom_left)
        bottom_right = self.find_min_distance(quoc_huy, list_bottom_right)
        bbox_list = [top_left, top_right, bottom_left, bottom_right, quoc_huy]
        return bbox_list
        
    def adding_corner_missing(self, bbox_list):
        ####box = [top_leftx, top_lefty, w, h, confidence, class]
        b1, b2, b3, quoc_huy = bbox_list
        dis1 = self.compute_distance(b1,b2)
        dis2 = self.compute_distance(b2,b3)
        dis3 = self.compute_distance(b3,b1)
        max_distance = max(dis1, dis2, dis3)
        p1, p2 = None, None
        q1, q2 = None, None
        if max_distance == dis1:
            p1, p2 = b1, b2
            q1 = b3
        elif max_distance == dis2:
            p1, p2 = b2, b3
            q1 = b1
        else:
            p1, p2 = b3, b1
            q1 = b2
        center = (int((p1[0] + p1[2]/2 + p2[0] + p2[2]/2)/2), int((p1[1] + p1[3]/2 + p2[1] + p2[3]/2)/2))
        q2 = [int(2 * center[0] - q1[0] - q1[2]/2), int(2 * center[1] - q1[1] - q1[3]/2)]
        q2 += [1,1]
        bbox_list = [p1, p2, q1, q2, quoc_huy]
        return bbox_list

    def evaluate_center_corners(self, bbox_list):
        top_left_box, top_right_box, bottom_left_box, bottom_right_box, id_card = bbox_list
        #margin = 5
        x_top_left = int((top_left_box[0] + top_left_box[2]/2))
        y_top_left = int((top_left_box[1] + top_left_box[3]/2))
        top_left = [x_top_left, y_top_left]

        x_top_right = int((top_right_box[0] + top_right_box[2]/2 ))
        y_top_right = int((top_right_box[1] + top_right_box[3]/2))
        top_right = [x_top_right, y_top_right]

        x_bottom_left = int((bottom_left_box[0] + bottom_left_box[2]/2))
        y_bottom_left = int((bottom_left_box[1] + bottom_left_box[3]/2) )
        bottom_left = [x_bottom_left, y_bottom_left]

        x_bottom_right = int((bottom_right_box[0] + bottom_right_box[2]/2) )
        y_bottom_right = int((bottom_right_box[1] + bottom_right_box[3]/2) )
        bottom_right = [x_bottom_right, y_bottom_right]

        corners = list([top_left, top_right, bottom_left, bottom_right])
        
        return corners



    def detect_4points(self, image):
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

        check_classes_dupplicate = []
        image_copy = image.copy()
        for i in indices:
            i = i[0]
            box = bboxes_pred[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            score = confidences[i]
            label = class_ids[i]
            check_classes_dupplicate.append(int(label))
            bboxes_output.append([x,y,w,h,score,label])
            p1 = (int(x),int(y))
            p2 = (int(x + w), int(y+h))
            cv2.rectangle(image_copy,p1, p2, (0,255,0), 2)
            cv2.putText(image_copy, self.classes[int(label)], p1, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255),1, cv2.LINE_AA)
        cv2.imshow("image_cropper", image_copy)
        cv2.imwrite('result/' + 'cropper_result.jpg', image_copy)
        # cv2.waitKey()

        classes_id = []
        for b in bboxes_output:
            classes_id.append(b[5])
        idx = np.argsort(classes_id)
        corners = 0
        return_code = 0
        
        ##check xem co nhan dien duoc quoc huy hay khong
        if 4 not in classes_id:
            print("Khong nhan dien duoc quoc huy, hay thu voi anh khac !!!")
            return return_code, corners
        if len(bboxes_output) == 5:
            top_left_box = bboxes_output[idx[0]]
            top_right_box = bboxes_output[idx[1]]
            bottom_left_box = bboxes_output[idx[2]]
            bottom_right_box = bboxes_output[idx[3]]
            id_card = bboxes_output[idx[4]]

            bbox_list = [top_left_box, top_right_box, bottom_left_box, bottom_right_box, id_card]
            if len(check_classes_dupplicate) == len(set(check_classes_dupplicate)) and \
                                    self.check_4points_location(bbox_list):
                corners = self.evaluate_center_corners(bbox_list)
            else:
                bbox_list = self.locate_4points(bbox_list)
                corners = self.evaluate_center_corners(bbox_list)
            return_code = 1   
            #return 1, corners
        
        elif len(bboxes_output) > 5:
            try:
                bbox_list = []
                for i in idx:
                    bbox_list.append(bboxes_output[i])
                bbox_list = self.erase_corner_redundant(bbox_list)
                bbox_list = self.locate_4points(bbox_list)
                corners = self.evaluate_center_corners(bbox_list)
                return_code=1
                #return 1, corners
            except:
                print("ERROR: Co loi khi xu ly truong hop nhan dien duoc nhieu hon 4 goc!!!!!!")
                return_code = 0
                # return 0, corners
        
        elif len(bboxes_output) == 4:
            try:
                bbox_list = []
                for i in idx:
                    bbox_list.append(bboxes_output[i])
                
                bbox_list = self.adding_corner_missing(bbox_list)
                bbox_list = self.locate_4points(bbox_list)
                corners = self.evaluate_center_corners(bbox_list)
                return_code = 1
                # return 1, corners
            except:
                print("ERROR: Co loi khi xu ly truong hop nhan dien duoc it hon 4 goc")
                return_code = 0
                #return 0, corners
        else:
            print("Mo hinh nhan duoc <= 2 goc, thu detect information truc tiep !!")
            return_code = 2
        return return_code, corners

    def crop_and_align_image(self,image):
        ###detect 4 corner and quoc_huy
        return_code, corners = self.detect_4points(image)
        if return_code == 0 or return_code == 2:
            return return_code, corners
        ## can chinh lai 4 goc de cat anh khong bi mat thong tin
        epsilon = 0.000001
        [p0,p1,p2,p3] = corners
        [x0,y0], [x1,y1],[x2,y2],[x3,y3] = p0,p1,p2,p3
        y30 = y3 - y0
        x30 = x3 - x0
        y21 = y2 - y1
        x21 = x2 - x1
        a,b,c,d = y2, y21/(x21+epsilon), x2 - x3, x30/(y30+epsilon)
        y = (a-b*c-b*d*y3)/(1-b*d + epsilon)
        x = x3 - d*(y3-y)
        # for i,p in enumerate(corners):
        #     cv2.circle(image, (p[0],p[1]), (i+1)*3, (0,255,0),-1)
        # cv2.circle(image, (int(x), int(y)), 5, (0,0,255), -1)
        #cv2.imshow("image", image)
        center = [x,y]
        new_corners = []
        
        for point in corners:
            ratio = abs((point[0] - center[0])/(point[1] - center[1] + epsilon))
            a,b=0,0
            dis = np.sqrt(((point[0] - center[0]) ** 2) + ((point[1] - center[1]) ** 2))
            margin = dis / 10
            if ratio >=1:
                a = int(margin)
                b = int(a /ratio)
            else:
                b = int(margin)
                a = int(b * ratio)
                
            if center[0]>=point[0] and center[1]>=point[1]:
                new_corners.append([point[0] - a, point[1]-b])
            elif center[0]<point[0] and center[1]>=point[1]:
                new_corners.append([point[0] + a, point[1]-b])
            elif center[0]>=point[0] and center[1]<point[1]:
                new_corners.append([point[0] - a, point[1]+b])
            elif center[0]<point[0] and center[1]<point[1]:
                new_corners.append([point[0] + a, point[1]+b])

        # for i,p in enumerate(new_corners):
        #     cv2.circle(image, (p[0],p[1]), (i+1)*3, (0,255,0),-1)
        # cv2.circle(image, (int(x), int(y)), 5, (0,0,255), -1)
        # cv2.imshow("image", image)
        # cv2.waitKey()

        ####align image ==================================================
        [top_left, top_right,bottom_left, bottom_right] = new_corners
        pts = np.array([top_left, top_right, bottom_right,bottom_left]).astype('float32')

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        width_a = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
        width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        height_a = np.sqrt(((bottom_right[0] - top_right[0]) ** 2) + ((bottom_right[1] - top_right[1]) ** 2))
        height_b = np.sqrt(((bottom_left[0] - top_left[0]) ** 2) + ((bottom_left[1] - top_left[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        aligned_image = cv2.resize(warped, (660, 416))
        return return_code, aligned_image


if __name__ == "__main__":
    type_img = ['jpg', 'png']
    image_folder = 'test_images/'
    #save_dir = 'aligned_image_after_processing/'
    images = os.listdir(image_folder)
    cropper = Cropper()
    print(len(images))
    for image_file in images:
        if (image_file[-3:] in type_img):# and (image_file[-7:-4] == '572' or image_file[-7:-4] == '106'):
            start = time.time()
            path = image_folder + image_file
            image = cv2.imread(path)
            H, W = image.shape[:2]
            image_resized = cv2.resize(image, (416, int(416 * H/W)))

            cv2.imshow("raw_image", image_resized)
            return_code, aligned_image = cropper.crop_and_align_image(image)
            if return_code != 1:
                continue
            print("Time processing: ", time.time() - start)
            #cv2.imwrite(image_folder + 'cropper.jpg', aligned_image)
            cv2.imshow("img_detected", aligned_image)
            cv2.waitKey(0)
            
    cv2.destroyAllWindows()
    



