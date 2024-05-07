#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int64MultiArray

class TrafficLightDetection() :

    def __init__(self) :
        rospy.init_node("traffic_light_detection")
        self.bridge = CvBridge()

        self.cv2_image = []
        self.image_sub   = rospy.Subscriber("/image_jpeg/compressed_traffic", CompressedImage, self.imgCB)
        self.traffic_light_pub = rospy.Publisher("/traffic_light", Int64MultiArray, queue_size=1)

        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            if len(self.cv2_image) != 0:
                
                y_roi_start = 210
                y_roi_end   = 430
                x_roi_start = 490
                x_roi_end   = 890

                cv2_image_cropped = self.cv2_image.copy()
                cv2_image_cropped = cv2_image_cropped[y_roi_start:y_roi_end, x_roi_start:x_roi_end,  :]
                
                hsv_red   = cv2.cvtColor(cv2_image_cropped, cv2.COLOR_BGR2HSV)
                hsv_green = cv2.cvtColor(cv2_image_cropped, cv2.COLOR_BGR2HSV)

                # h, s, v = cv2.split(hsv_red)

                lower_red_1 = np.array([0, 160, 125])
                upper_red_1 = np.array([3, 255, 255]) 
                red_mask_1  = cv2.inRange(hsv_red, lower_red_1, upper_red_1)

                lower_red_2 = np.array([165, 160, 125])
                upper_red_2 = np.array([180, 255, 255]) 
                red_mask_2  = cv2.inRange(hsv_red, lower_red_2, upper_red_2)

                red_mask = red_mask_1 + red_mask_2

                lower_green = np.array([60, 200, 120]) #(40,50,50) (60,50,50)
                upper_green = np.array([70, 255, 255]) #70. 90
                green_mask  = cv2.inRange(hsv_green, lower_green, upper_green)

                red_pixel_counts   = np.count_nonzero(red_mask)
                green_pixel_counts = np.count_nonzero(green_mask)
                
                msg = Int64MultiArray()
                msg.data = [red_pixel_counts, green_pixel_counts]
                self.traffic_light_pub.publish(msg)
                # print(msg)

                # cv2.imshow('h', h)
                # cv2.imshow('s', s)
                # cv2.imshow('v', v)
                # cv2.imshow("cropped", cv2_image_cropped)
                # cv2.imshow("original", self.cv2_image)
                # cv2.imshow/
                # cv2.imshow('green', green_mask)
                # cv2.imshow('red', red_mask)

                cv2.waitKey(1)

                rate.sleep()
            else:
                # print("Traffic Light Detection is not working")
                pass

    def imgCB(self, msg) :
        self.cv2_image = self.bridge.compressed_imgmsg_to_cv2(msg)                                                                                  


if __name__ == "__main__":
    try:
        traffic_light_detection = TrafficLightDetection()
    except:
        pass

 
