#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import rospy 
from morai_msgs.msg import GetTrafficLightStatus, CtrlCmd, GPSMessage
from std_msgs.msg import Float64, Int64
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Imu
from geometry_msgs.msg import  Vector3
from move_base_msgs.msg import MoveBaseActionResult
from cv_bridge import CvBridge
from pyproj import Proj, transform
from nav_msgs.msg import Odometry
from math import pi, sqrt, atan2, radians
from lidar_object_detection.msg import ObjectInfo

from slidewindow_first_blackout import SlideWindow1
from slidewindow_second_blackout import SlideWindow2

import tf
import time
import cv2
import numpy as np

class PID():
  def __init__(self,kp,ki,kd):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.p_error = 0.0
    self.i_error = 0.0
    self.d_error = 0.0

  def pid_control(self, cte):
    self.d_error = cte-self.p_error
    self.p_error = cte
    self.i_error += cte

    return self.kp*self.p_error + self.ki*self.i_error + self.kd*self.d_error

class  LaneDetection:
    def __init__(self):
        rospy.init_node("lane_detection_node")

        self.ctrl_cmd_pub = rospy.Publisher('/lane_ctrl_cmd', CtrlCmd, queue_size=1)

        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.camCB)
        rospy.Subscriber("/heading", Float64, self.headingCB)
        rospy.Subscriber("/current_waypoint", Int64, self.waypointCB)
        rospy.Subscriber("/object_info", ObjectInfo, self.lidarObjectCB)
        #-------------------------------------------------------------------------------------- #
        self.proj_UTM = Proj(proj='utm', zone = 52, elips='WGS84', preserve_units=False)
        self.bridge = CvBridge()
        self.ctrl_cmd_msg = CtrlCmd()
        # self.status_msg   = EgoStatus()
        self.traffic_msg = GetTrafficLightStatus()

        self.ctrl_cmd_msg.longlCmdType = 2
        self.is_gps = True
        self.is_imu = True

        self.slidewindow_1 = SlideWindow1()
        self.slidewindow_2 = SlideWindow2()
        self.traffic_flag = 0
        self.prev_signal = 0
        self.signal = 0
        self.stopline_flag = 0
        self.img = []
        self.warped_img = []
        self.grayed_img = []
        self.out_img = []
        self.yellow_img = []
        self.white_img = []
        self.img_hsv = []
        self.h = []
        self.s = []
        self.v = []
        self.bin_img = []
        self.left_indices = []
        self.right_indices = []
        self.lidar_obstacle_info = []
        self.gt_heading = 0
        
        self.heading =0
        self.x_location = 640
        self.last_x_location = 640

        self.prev_center_index = 640
        self.center_index = 640
        self.standard_line = 640
        self.degree_per_pixel = 0
        self.avoid_x_location = 640

        self.current_lane = 2
        self.avoid_status = 'lanekeeping'
        self.sliding_window_select_line = 'Right'
        
        self.current_waypoint = 0

        rate = rospy.Rate(20)  # hz 
        while not rospy.is_shutdown():
            # self.getEgoStatus()

            if len(self.img)!= 0:

                y, x = self.img.shape[0:2]
                self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(self.img_hsv)
                yellow_lower = np.array([15, 110, 80])
                yellow_upper = np.array([40, 255, 255])
                self.yellow_range = cv2.inRange(self.img_hsv, yellow_lower, yellow_upper)
                
                white_lower = np.array([0, 0, 100])
                white_upper = np.array([179, 64, 255])
                self.white_range = cv2.inRange(self.img_hsv, white_lower, white_upper)
                
                combined_range = cv2.bitwise_or(self.yellow_range, self.white_range)
                filtered_img = cv2.bitwise_and(self.img, self.img, mask=combined_range)
                
                src_point1 = [375, 360]      # 왼쪽 아래
                src_point2 = [575, 201]
                src_point3 = [x-575, 201]
                src_point4 = [x - 375, 360]      # 오른쪽 아래
                # src_point1 = [0, 670]      # 왼쪽 아래
                # src_point2 = [585, 414]
                # src_point3 = [x-585, 414]
                # src_point4 = [x , 670]  
                src_points = np.float32([src_point1,src_point2,src_point3,src_point4])
                
                dst_point1 = [x//8, 720]    # 왼쪽 아래
                dst_point2 = [x//8, 0]      # 왼쪽 위
                dst_point3 = [x//8*7, 0]    # 으론쪽 위
                dst_point4 = [x//8*7, 720]  # 오른쪽 아래
                dst_points = np.float32([dst_point1,dst_point2,dst_point3,dst_point4])
                
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                self.warped_img = cv2.warpPerspective(filtered_img, matrix, [x,y])
                self.grayed_img = cv2.cvtColor(self.warped_img, cv2.COLOR_BGR2GRAY)
                
                
                # 이미지 이진화
                self.bin_img = np.zeros_like(self.grayed_img)
                self.bin_img[self.grayed_img > 150] = 1



                if self.x_location == None :
                    self.x_location = self.last_x_location
                else :
                    self.last_x_location = self.x_location
                


                try:
                    cv2.rectangle(self.warped_img, [self.left_indices[0], 0], [self.left_indices[-1], y], [255,0,0], 2)
                except:
                    pass
                try:
                    cv2.rectangle(self.warped_img, [self.right_indices[0], 0], [self.right_indices[-1], y], [255,0,0], 2)
                except:
                    pass
                try:
                    cv2.rectangle(self.warped_img, [self.center_index-3, 238], [self.center_index+3, 242], [0,0,255], 3)
                except:
                    pass
                    
                

                if self.current_waypoint < 2250: # first black out
                    
                    self.out_img, self.x_location, _ = self.slidewindow_1.slidewindow(self.bin_img)

                else:
                    self.out_img, self.x_location, _ = self.slidewindow_2.slidewindow(self.bin_img)
                
                if self.x_location == None :
                    self.x_location = self.last_x_location
                else :
                    self.last_x_location = self.x_location
                

                self.standard_line = x//2
                self.degree_per_pixel = 1/x
                self.prev_center_index = self.center_index
                

                # print(self.steer_msg.data)
                # cv2.imshow("img", self.img)
                # cv2.imshow("out_img", self.out_img)
                # cv2.waitKey(1)
        
                #0.001, 0.001, 0.01
                #0.033, 0.003, 0.015
                #0.035, 0.003, 0.015
                pid = PID(0.02, 0.003, 0.010)
                # pid = PID(0.03, 0.005, 0.025)
                # pid = PID(0.01, 0.015, 0.001)
                ################################################################################
                self.center_index = self.x_location
                
                angle = pid.pid_control(self.center_index - 640)
                motor_msg = 15
                # self.gt_heading = -40
                ################################################################################
                # if self.avoid_status == 'lanekeeping':
                    
               
                #     if self.is_something_front(self.lidar_obstacle_info) is True:
                #         self.avoid_status = 'avoiding'
                #     else:
                #         pass
                # else:
                #     pass

                
                # print("current_lane: ", self.current_lane)
                # if self.current_lane == 3:

                #     if self.avoid_status == 'avoiding':
                #         # gt_heading 52
                #         # if self.gt_heading >= 0:
                        
                #         if self.heading <= self.gt_heading + 17:
                #             self.avoid_x_location -= 1
                #             self.center_index = self.avoid_x_location
                #             angle = self.center_index - 640
                #             motor_msg = 15
                #             # print("!!!!!!!!")
                     
                #         elif self.heading > self.gt_heading + 17:
                #             self.avoid_x_location  = 640
                #             self.avoid_status = 'returning'

                #     elif self.avoid_status == 'returning':
                #         if self.heading > self.gt_heading:
                #             self.avoid_x_location += 0.5
                #             self.center_index = self.avoid_x_location
                #             angle = self.center_index - 640
                #             motor_msg = 15
                #         elif self.heading <= self.gt_heading:
                #             self.current_lane -= 1
                #             self.avoid_status = 'lanekeeping'
                #     else:
                #         pass
                    

            
                # else:
                #     if self.avoid_status == 'avoiding':
                    
                #         if self.heading >= self.gt_heading - 18:
                #             self.avoid_x_location += 1
                #             self.center_index = self.avoid_x_location
                #             angle = self.center_index - 640
                #             motor_msg = 15

                #         elif self.heading < self.gt_heading - 18:
                #             self.avoid_x_location  = 640
                #             self.avoid_status= 'returning'

                #     elif self.avoid_status == 'returning':
                #         if self.heading < self.gt_heading-5:
                #             self.avoid_x_location -= 0.5
                #             self.center_index = self.avoid_x_location
                #             angle = self.center_index - 640
                #             motor_msg = 15

                #         elif self.heading >= self.gt_heading-5:
                #             self.current_lane += 1
                #             self.avoid_status = 'lanekeeping'
                    
                #     else:
                #         pass

                
                # print(self.avoid_x_location)
                # print(f'self.start_avoid: {self.start_avoid}')
                # print(f'self.current_lane: {self.current_lane}')

                # print(f'self.avoid_status: {self.avoid_status}')

                servo_msg = -radians(angle)
                # print(f'self.heading {self.heading}')
                # print(f'self.gt_heading: {self.gt_heading}')
                # print(f'self.is_gps: {self.is_gps}')
                # print(self.lidar_obstacle_info)
                # print(servo_msg)

                self.publishCtrlCmd(motor_msg, servo_msg, 0)

            rate.sleep()

            
###################################################################### Call Back ######################################################################

    def publishCtrlCmd(self, motor_msg, servo_msg, brake_msg):
        self.ctrl_cmd_msg.velocity = motor_msg
        self.ctrl_cmd_msg.steering = servo_msg
        self.ctrl_cmd_msg.brake = brake_msg
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

    def camCB(self, msg):
        self.img = self.bridge.compressed_imgmsg_to_cv2(msg)

    def lidarObjectCB(self, msg):
        
        self.lidar_obstacle_info = [[] for i in range(msg.objectCounts)]
        for i in range(msg.objectCounts):
            self.lidar_obstacle_info[i] = [msg.centerX[i], msg.centerY[i], msg.centerZ[i], msg.lengthX[i], msg.lengthY[i], msg.lengthZ[i]]
        # print(self.lidar_obstacle_info)

    def headingCB(self, msg):
        self.heading = msg.data

    def waypointCB(self, msg):
        self.current_waypoint = msg.data

    
##################################################Function##########################################
    
    # def ChangeLane(self, current_lane, heading, mode):
        
    #     if current_lane == 3:


    def is_something_front(self, obstacle_info):# self.lidar_obstacle_info[i] = [msg.centerX[i], msg.centerY[i], msg.centerZ[i], msg.lengthX[i], msg.lengthY[i], msg.lengthZ[i]]
        for i in range(len(obstacle_info)):
            if 5.0 <= obstacle_info[i][0]< 20.0 and abs(obstacle_info[i][1]) < 1:#and self.avoid_status == 'lanekeeping':
                return True
            else: return False 

        
    

if __name__ == "__main__":
    try: 
        lane_detection_node = LaneDetection()
    except rospy.ROSInterruptException:
        pass
