#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import rospy 
from morai_msgs.msg import GetTrafficLightStatus, CtrlCmd, GPSMessage
from std_msgs.msg import Float64, Int64, Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Imu
from geometry_msgs.msg import  Vector3
from move_base_msgs.msg import MoveBaseActionResult
from cv_bridge import CvBridge
from pyproj import Proj, transform
from nav_msgs.msg import Odometry
from math import pi, sqrt, atan2, radians
from lidar_object_detection.msg import ObjectInfo
from lidar_cam_fusion.msg import Float64Array2D

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
        self.stopline_flag_pub = rospy.Publisher('/stopline_flag', Bool, queue_size=1)

        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.camCB)
        rospy.Subscriber("/heading", Float64, self.headingCB)
        rospy.Subscriber("/current_waypoint", Int64, self.waypointCB)
        # rospy.Subscriber("/object_info_static", ObjectInfo, self.lidarObjectCB)
        rospy.Subscriber("/fusion_result", Float64Array2D, self.fusionResultCB)
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

        self.up_hist_end_line = 400
        self.down_hist_start_line = 400

        self.fusion_result = []

        rate = rospy.Rate(20)  # hz 
        while not rospy.is_shutdown():
            # self.getEgoStatus()

            if len(self.img)!= 0:

                y, x = self.img.shape[0:2]
                self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(self.img_hsv)
                yellow_lower = np.array([0, 100, 180])
                yellow_upper = np.array([23, 180, 230])
                self.yellow_range = cv2.inRange(self.img_hsv, yellow_lower, yellow_upper)
                
                # white_lower = np.array([0, 0, 70])
                # white_upper = np.array([110, 80, 255]) #np.array([179, 64, 255])
                # self.white_range = cv2.inRange(self.img_hsv, white_lower, white_upper)
                white_lower_bound1 = np.array([0, 25, 60])
                white_upper_bound1 = np.array([110, 80, 90])
                white_mask1 = cv2.inRange(self.img_hsv, white_lower_bound1, white_upper_bound1)
                
                white_lower_bound2 = np.array([0, 25, 150])
                white_upper_bound2 = np.array([110, 80, 250])
                white_mask2 = cv2.inRange(self.img_hsv, white_lower_bound2, white_upper_bound2)

                self.white_range = cv2.bitwise_or(white_mask1, white_mask2)
                
                combined_range = cv2.bitwise_or(self.yellow_range, self.white_range)

                gray_lower_bound = np.array([5, 0, 5])
                gray_upper_bound = np.array([30, 100, 215])
                tmp = cv2.inRange(self.img_hsv, gray_lower_bound, gray_upper_bound)

                result = cv2.subtract(combined_range, tmp)
                filtered_img = cv2.bitwise_and(self.img, self.img, mask=result)


                # filtered_img = cv2.bitwise_(filtered_img, filtered_img, mask=tmp)

                # src_point1 = [0, 720]      # 왼쪽 아래
                # src_point2 = [565, 400]
                # src_point3 = [x-565, 400]
                # src_point4 = [x , 720]  

                left_margin = 510
                top_margin = 428
                src_point1 = [0, 720]      # 왼쪽 아래
                src_point2 = [left_margin, top_margin]
                src_point3 = [x-left_margin, top_margin]
                src_point4 = [x , 720]  

                src_points = np.float32([src_point1,src_point2,src_point3,src_point4])
                
                # dst_point1 = [x//8, 720]    # 왼쪽 아래
                # dst_point2 = [x//8, 0]      # 왼쪽 위
                # dst_point3 = [x//8*7, 0]    # 으론쪽 위
                # dst_point4 = [x//8*7, 720]  # 오른쪽 아래

                dst_point1 = [x//4, 720]    # 왼쪽 아래
                dst_point2 = [x//4, 0]      # 왼쪽 위
                dst_point3 = [x//4*3, 0]    # 으론쪽 위
                dst_point4 = [x//4*3, 720]  # 오른쪽 아래

                dst_points = np.float32([dst_point1,dst_point2,dst_point3,dst_point4])
                
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                self.warped_img = cv2.warpPerspective(filtered_img, matrix, [x,y])
                self.grayed_img = cv2.cvtColor(self.warped_img, cv2.COLOR_BGR2GRAY)
                
                # 이미지 이진화
                self.bin_img = np.zeros_like(self.grayed_img)

                # grayed_mask1 = cv2.inRange(self.grayed_img, 60, 80)
                # grayed_mask2 = cv2.inRange(self.grayed_img, 150, 255)
                # grayed_combined_mask = cv2.bitwise_or(grayed_mask1, grayed_mask2)
                self.bin_img[self.grayed_img > 60] = 1
                # self.bin_img[self.grayed_img > 150] = 1 #  > 150
                
                histogram_y = np.sum(self.bin_img, axis=1)
                # print(f"histogram_y: {histogram_y}")


                up_hist = histogram_y[0:self.up_hist_end_line]
                down_hist = histogram_y[self.down_hist_start_line:]

                stopline_indices = np.where(down_hist > 600)[0] + self.down_hist_start_line # 500
                

                try:
                    stopline_threshold = 10 # 10
                    stopline_diff = stopline_indices[-1] - stopline_indices[0]
                    if stopline_threshold < stopline_diff:
                        self.stopline_flag = True
                        # cv2.rectangle(self.warped_img, [0, stopline_indices[0]], [x, stopline_indices[-1]], [0,255,0], 3)
                    else:
                        self.stopline_flag = False
                except:
                    self.stopline_flag = False

                if self.x_location == None :
                    self.x_location = self.last_x_location
                else :
                    self.last_x_location = self.x_location

                    
                
                if self.current_waypoint < 2250: # first black out
                    self.out_img, self.x_location, _ = self.slidewindow_1.slidewindow(self.bin_img)
                    pid = PID(0.020, 0.01, 0.010)

                else:
                    self.out_img, self.x_location, _ = self.slidewindow_2.slidewindow(self.bin_img)
                    pid = PID(0.028, 0.005, 0.010)



                try:
                    cv2.rectangle(self.out_img, [0, stopline_indices[0]], [x, stopline_indices[-1]], [0, 0, 255], 3)
                except:
                    pass

                if self.x_location == None :
                    self.x_location = self.last_x_location
                else :
                    self.last_x_location = self.x_location
                
                

                self.standard_line = x//2
                self.degree_per_pixel = 1/x
                self.prev_center_index = self.center_index
                self.center_index = self.x_location
                #0.015, 0.003, 0.010
                # pid = PID(0.015, 0.003, 0.010)
                motor_msg = 15

                angle = pid.pid_control(self.center_index - 640)
                
                if self.current_waypoint >= 3030:
                    motor_msg = 22

                    self.gt_heading = -41

                    if self.avoid_status == 'lanekeeping':
                        # self.gt_heading = self.heading
                        if self.is_something_front(self.fusion_result) is True:
                            self.avoid_status = 'avoiding'

                        else:
                            pass
                    else:
                        motor_msg = 22
                    
                    if self.current_lane == 3:

                        if self.avoid_status == 'avoiding':
                            # gt_heading 52
                            # if self.gt_heading >= 0:
                            
                            if self.heading <= self.gt_heading + 12:  # 10.5 
                                self.avoid_x_location -= 1.0 # 1.0
                                self.center_index = self.avoid_x_location
                                angle = self.center_index - 640
                                # print("111")
                        
                            elif self.heading > self.gt_heading + 12:  # 10.5
                                self.avoid_x_location  = 640 #######
                                self.avoid_status = 'returning'
                                # print("222")

                        elif self.avoid_status == 'returning':
                            if self.heading > self.gt_heading + 5:
                                self.avoid_x_location += 0.5 # 0.5
                                self.center_index = self.avoid_x_location
                                angle = self.center_index - 640

                            elif self.heading <= self.gt_heading + 5:
                                self.current_lane -= 1
                                self.avoid_status = 'lanekeeping'
                        else:
                            pass
                        

                
                    else:
                        if self.avoid_status == 'avoiding':
                        
                            if self.heading >= self.gt_heading - 12:
                                self.avoid_x_location += 1.0 #1
                                self.center_index = self.avoid_x_location
                                angle = self.center_index - 640
                                # print('angle______1:', angle)

                            elif self.heading < self.gt_heading - 12:
                                self.avoid_x_location  = 640
                                self.avoid_status= 'returning'

                        elif self.avoid_status == 'returning':
                            if self.heading < self.gt_heading - 5:
                                self.avoid_x_location -= 0.5 #0.5
                                self.center_index = self.avoid_x_location
                                angle = self.center_index - 640

                            elif self.heading >= self.gt_heading - 5:
                                self.current_lane += 1
                                self.avoid_status = 'lanekeeping'
                        
                        else:
                            pass
                            
                
                
                # print(f'self.avoid_status: {self.avoid_status}')
                # print(f'self.heading: {self.heading}')

                
                
                
                servo_msg = -radians(angle)
              
                self.stopline_flag_pub.publish(self.stopline_flag)
                self.publishCtrlCmd(motor_msg, servo_msg, 0)

                # cv2.imshow("img", self.img)
                # cv2.imshow("h", h)
                # cv2.imshow("s", s)
                # cv2.imshow("v", v)
                # cv2.imshow("result", result)
                # cv2.imshow("white_range", self.white_range)
                # cv2.imshow("yellow_range", self.yellow_range)
                # cv2.imshow("combined_range", combined_range)
                # cv2.imshow("filtered_img", filtered_img)
                # cv2.imshow("warped_img", self.warped_img)
                # cv2.imshow("grayed_img", self.grayed_img)
                # cv2.imshow("bin_img", self.bin_img)
                cv2.imshow("out_img", self.out_img)
                cv2.waitKey(1)

            rate.sleep()

            
###################################################################### Call Back ######################################################################

    def publishCtrlCmd(self, motor_msg, servo_msg, brake_msg):
        self.ctrl_cmd_msg.velocity = motor_msg
        self.ctrl_cmd_msg.steering = servo_msg
        self.ctrl_cmd_msg.brake = brake_msg
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

    def camCB(self, msg):
        self.img = self.bridge.compressed_imgmsg_to_cv2(msg)

    # def lidarObjectCB(self, msg):
        
    #     self.lidar_obstacle_info = [[] for i in range(msg.objectCounts)]
    #     for i in range(msg.objectCounts):
    #         self.lidar_obstacle_info[i] = [msg.centerX[i], msg.centerY[i], msg.centerZ[i], msg.lengthX[i], msg.lengthY[i], msg.lengthZ[i]]
    #     # print(self.lidar_obstacle_info)

    def fusionResultCB(self, msg):
        self.fusion_result = [list(bbox.bbox)[4:8] for bbox in msg.bboxes]

    def headingCB(self, msg):
        self.heading = msg.data

    def waypointCB(self, msg):
        self.current_waypoint = msg.data

    
##################################################Function##########################################
    
    # def ChangeLane(self, current_lane, heading, mode):
        
    #     if current_lane == 3:


    def is_something_front(self, obstacle_info):# self.lidar_obstacle_info[i] = [msg.centerX[i], msg.centerY[i], msg.centerZ[i], msg.lengthX[i], msg.lengthY[i], msg.lengthZ[i]]
        for i in range(len(obstacle_info)):
            # if 5.0 <= obstacle_info[i][0]< 20.0 and abs(obstacle_info[i][1]) < 0.8: #and self.avoid_status == 'lanekeeping':
            if abs(obstacle_info[i][1]) < 1.0: # 0.7
                return True
            else: 
                return False 

   

        
    

if __name__ == "__main__":
    try: 
        lane_detection_node = LaneDetection()
    except rospy.ROSInterruptException:
        pass
