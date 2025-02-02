#!/usr/bin/python3
#-*- encoding: utf-8 -*-
import os.path as ops
# import json
import numpy as np
import torch
import cv2
import time
import math
import os
import matplotlib.pylab as plt
import sys
import rospy
import imageio
from tqdm import tqdm
from dataset.dataset_utils import TUSIMPLE
from Lanenet.model2 import Lanenet
from utils.evaluation import gray_to_rgb_emb, process_instance_embedding
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
# from race.msg import drive_values
from driving_es import *
from slidewindow2 import *
from std_msgs.msg import Int32
from std_msgs.msg import Bool
from utils.lane import LaneEval


# drive_values_pub= rospy.Publisher('control_value', drive_values, queue_size = 1)


bridge = CvBridge()
image = np.empty(shape=[0])

# rospy.init_node("lane")

torch.backends.cudnn.enabled = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'

slidewindow = SlideWindow()
x_location = 320
last_x_location = 320
is_detected = True
current_lane = "LEFT"
last_angle = 0

def image_callback(img_data):
	global bridge
	global image
	image = bridge.compressed_imgmsg_to_cv2(img_data, "rgb8")
        

# Load the Model
model_path = "/home/foscar/ISEV_2024/src/lane_pkg/src/TUSIMPLE/Lanenet_output/lanenet_epoch_39_batch_8.model" #내동영상으로 바꿔준다
LaneNet_model = Lanenet(2, 4)
LaneNet_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
LaneNet_model.to(device)

def save_last_angle(angle):
     global last_angle
     last_angle = angle

def inference(gt_img_org):

    # BGR 순서

    org_shape = gt_img_org.shape
    gt_image = cv2.resize(gt_img_org, dsize=(512, 256), interpolation=cv2.INTER_LINEAR)
    gt_image = gt_image / 127.5 - 1.0
    gt_image = torch.tensor(gt_image, dtype=torch.float)
    gt_image = np.transpose(gt_image, (2, 0, 1))
    gt_image = gt_image.to(device)
    # lane segmentation 
    binary_final_logits, instance_embedding = LaneNet_model(gt_image.unsqueeze(0))
    binary_final_logits, instance_embedding = binary_final_logits.to('cpu'), instance_embedding.to('cpu') 
    binary_img = torch.argmax(binary_final_logits, dim=1).squeeze().numpy()
    binary_img[0:85,:] = 0 #(0~85행)을 무시 - 불필요한 영역 제거

    # 차선 클러스터링, 색상 지정
    rbg_emb, cluster_result = process_instance_embedding(instance_embedding, binary_img,distance=1.5, lane_num=3)
    rbg_emb = cv2.resize(rbg_emb, dsize=(org_shape[1], org_shape[0]), interpolation=cv2.INTER_LINEAR)
    a = 0.1
    frame = a * gt_img_org[..., ::-1] / 255 + rbg_emb * (1 - a)
    frame = np.rint(frame * 255)
    frame = frame.astype(np.uint8)
    #차선 검출 신뢰도_____________________________________
    # 진짜 차선 정보가 담긴 json파일과 prediction한 json파일을 비교해서 신뢰도 평가
    # json_pred = [json.loads(line) for line in open('pred.json').readlines()]
    # json_gt = [json.loads(line) for line in open('test_label_new.json')]


    #motor_info = drive_values()


    return frame



if __name__ == "__main__":
    image_sub = rospy.Subscriber("/image_jpeg/compressed", CompressedImage, image_callback)
    
    

    lx,ly,rx,ry = [200,] , [] , [428,] , []  #sliding_Window 함수 내에서 x_temp , y_temp 매개변수에 lx[0]과 rx[0]을 할당해주어야 하기에 초기화 합니다.


    rospy.init_node('lane_drive')
    # motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    #image_sub = rospy.Subscriber("/usb_cam/image_raw/",Image,img_callback)

    # print ("----- Xycar self driving -----")
    left_prob,right_prob = 0,0 #left_prob와 right_prob또한 slidingwindow 함수의 매개변수로 사용해야하기에 초기화합니다.

    x_location = 320
    last_x_location = 320
    is_detected = True
    current_lane = "LEFT"
    while not rospy.is_shutdown():
        try:
            show_img = image
            init_show_img(show_img)
            # print("@@@@@@@@@@@@@@@@@@@@@@")
            # cv2.imshow("lane_image", show_img)
            # cv2.waitKey(1)
            img_frame = show_img.copy() # img_frame변수에 카메라 이미지를 받아옵니다.   
            height,width,channel = img_frame.shape # 이미지의 높이,너비,채널값을 변수에 할당합니다. 
            # img_roi = img_frame[280:,0:]   # y좌표 0~320 사이에는 차선과 관련없는 이미지들이 존재하기에 노이즈를 줄이기 위하여 roi설정을 해주었습니다.
            
            img_filtered = color_filter(img_frame)   #roi가 설정된 이미지를 color_filtering 하여 흰색 픽셀만을 추출해냅니다. 
            cv2.imshow("just filetered", img_filtered)
            inferenced_filtered = inference(img_filtered)
            cv2.imshow("inferened and filetered", inferenced_filtered)
            img_warped = bird_eye_view(inferenced_filtered,width,height) # 앞서 구현한 bird-eye-view 함수를 이용하여 시점변환해줍니다. 


            _, L, _ = cv2.split(cv2.cvtColor(img_warped, cv2.COLOR_BGR2HLS))
            _, img_binary = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY) #color_filtering 된 이미지를 한번 더 이진화 하여 차선 검출의 신뢰도를 높였습니다. 

            # img_masked = region_of_interest(img_binary) #이진화까지 마친 이미지에 roi를 다시 설정하여줍니다.
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            out_img, x_location, current_lane= slidewindow.slidewindow(img_binary, is_detected)
            img_blended = cv2.addWeighted(out_img, 1, img_warped, 0.6, 0) # sliding window결과를 시각화하기 위하여 out_img와 시점변환된이미지를 merging 하였습니다. 
            # out_img, x_location, current_lane= slidewindow.slidewindow(img_blended, is_detected)
            
            cv2.imshow("CAM View", img_blended)
            cv2.waitKey(1)

            if x_location == None :
                x_location = last_x_location
                is_detected = False
            else :
                last_x_location = x_location
                is_detected = True




        except:
            pass

        rospy.sleep(0.05)
