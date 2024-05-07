#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import rospy 
from morai_msgs.msg import GetTrafficLightStatus, CtrlCmd, GPSMessage
from std_msgs.msg import Float64, Int32
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Imu
from geometry_msgs.msg import  Vector3
from move_base_msgs.msg import MoveBaseActionResult
from cv_bridge import CvBridge
from pyproj import Proj, transform
from nav_msgs.msg import Odometry
from math import pi, sqrt, atan2, radians
from lidar_object_detection.msg import ObjectInfo

from slidewindow_faster import SlideWindow

import tf
import time
import cv2
import numpy as np

# class EgoStatus:
#     def __init__(self):
#         self.position = Vector3()
#         self.heading = 0.0
#         self.velocity = Vector3()


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
        self.ctrl_cmd_pub = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)

        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.cam_CB)
        rospy.Subscriber("/heading", Float64, self.heading_CB)
        # Subscriber
        rospy.Subscriber("/gps", GPSMessage, self.gpsCB) ## Vehicle Status Subscriber 
        rospy.Subscriber("/imu", Imu, self.imuCB) ## Vehicle Status Subscriber
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

        self.slidewindow = SlideWindow()
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

        self.x_location = 640
        self.last_x_location = 640
        self.gt_heading = 0

        self.prev_center_index = 640
        self.center_index = 640
        self.standard_line = 640
        self.degree_per_pixel = 0
        self.avoid_x_location = 640

        self.current_lane = 2
        self.start_avoid = False
        self.is_slidewindow = False
        self.sliding_window_select_line = 'Right'
        

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
                
                # src_point1 = [128, 720]      # 왼쪽 아래
                # src_point2 = [528, 368]
                # src_point3 = [x-528, 368]
                # src_point4 = [x - 128, 720]      # 오른쪽 아래
                src_point1 = [0, 670]      # 왼쪽 아래
                src_point2 = [585, 414]
                src_point3 = [x-585, 414]
                src_point4 = [x , 670]  
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
                
                # print(f"following index: {self.center_index}")


                self.out_img, self.x_location, _ = self.slidewindow.slidewindow(self.bin_img)

                if self.x_location == None :
                    self.x_location = self.last_x_location
                else :
                    self.last_x_location = self.x_location
                

                self.standard_line = x//2
                self.degree_per_pixel = 1/x
                self.prev_center_index = self.center_index
                

                # if self.is_slidewindow is False:
                #     self.steer_msg.data = 0.5 + (self.center_index - self.standard_line) * self.degree_per_pixel
  
                # elif self.is_slidewindow is True:
                #     self.center_index = self.x_location
                #     # print(self.center_index)
                #     angle = pid.pid_control(self.center_index - 320)
                #     self.steer_msg.data = 0.5 + angle/100
 
                # print(self.steer_msg.data)
                cv2.imshow("img", self.img)
                cv2.imshow("out_img", self.out_img)
                cv2.waitKey(1)
                # self.motor_msg.data = self.speed * 300
                # self.publishCtrlCmd(self.motor_msg, self.steer_msg)
                #0.001, 0.001, 0.01
                pid = PID(0.033, 0.003, 0.015)

                self.center_index = self.x_location
                
                angle = pid.pid_control(self.center_index - 640)
                speed = 20

                
                if self.is_something_front(self.lidar_obstacle_info) is True:
                    self.start_avoid = True
                # else:
                #     if self.start_avoid is False:
                #         self.gt_heading = self.heading

                if self.start_avoid is True:
                    angle, speed = self.ChangeLane(self.current_lane, -40.0)
                

                print(f'self.start_avoid: {self.start_avoid}')



                steering = -radians(angle)
                print(f'self.heading {self.heading}')
                # print(f'self.is_gps: {self.is_gps}')

                if self.is_gps is False: # 최종 퍼블리시만 조건문으로 했지만 모든 이 코드를 묶어도 될 것 같다. 근데 모르겠다.
                # if self.is_gps is False and self.is_imu is True:
                    self.publishCtrlCmd(speed, steering, 0)


            rate.sleep()

            
###################################################################### Call Back ######################################################################
    def gpsCB(self, msg):
        if msg.status == 0: 
            self.is_gps = False
        else:
            self.xy_zone = self.proj_UTM(msg.longitude, msg.latitude)
            
            # self.tf_broadcaster.sendTransform((0, 0, 1.18),
            #                 tf.transformations.quaternion_from_euler(0, 0, 0),
            #                 rospy.Time.now(),
            #                 "gps",
            #                 "base_link")
            
            # self.tf_broadcaster.sendTransform((1.44, 0, 1.24),
            #                 tf.transformations.quaternion_from_euler(0, 0, 0),
            #                 rospy.Time.now(),
            #                 "velodyne",
            #                 "base_link")
            
            self.is_gps = True


    def imuCB(self, msg):
        self.quaternion_data = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        self.euler_data = tf.transformations.euler_from_quaternion((msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w))

        # self.tf_broadcaster.sendTransform((-0.08, 0.0, 1.18),
        #                 tf.transformations.quaternion_from_euler(0, 0, 0),
        #                 rospy.Time.now(),
        #                 "imu",
        #                 "base_link")

        self.is_imu = True

    def publishCtrlCmd(self, motor_msg, servo_msg, brake_msg):
        self.ctrl_cmd_msg.velocity = motor_msg
        self.ctrl_cmd_msg.steering = servo_msg
        self.ctrl_cmd_msg.brake = brake_msg
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

    def cam_CB(self, msg):
        self.img = self.bridge.compressed_imgmsg_to_cv2(msg)

    def lidarObjectCB(self, msg):
        
        self.lidar_obstacle_info = [[] for i in range(msg.objectCounts)]
        for i in range(msg.objectCounts):
            self.lidar_obstacle_info[i] = [msg.centerX[i], msg.centerY[i], msg.centerZ[i], msg.lengthX[i], msg.lengthY[i], msg.lengthZ[i]]
    def heading_CB(self, msg):
        self.heading = msg.data
##################################################Function##########################################
    
    def ChangeLane(self, current_lane, gt_heading):

         # gt_heading 52
        # print(f'gt_heading {gt_heading}')
        if self.gt_heading < 0:
            if current_lane == 3:
                if self.heading <= gt_heading + 18:
                    self.avoid_x_location -= 2
                    self.center_index = self.avoid_x_location
                    angle = self.center_index - 640
                    speed = 15

                elif self.heading < gt_heading + 18:
                    self.current_lane -= 1
                    self.avoid_x_location  = 640

                if self.heading > gt_heading:
                    self.avoid_x_location += 1
                    self.center_index = self.avoid_x_location
                    angle = self.center_index - 640
                    speed = 15
                elif self.heading <= gt_heading:
                    self.start_avoid = False
            
            else:
                if self.heading >= gt_heading - 18:
                    self.avoid_x_location += 2
                    self.center_index = self.avoid_x_location
                    angle = self.center_index - 640
                    speed = 15

                elif self.heading > gt_heading - 18:
                    self.current_lane += 1
                    self.avoid_x_location  = 640

                if self.heading < gt_heading:
                    self.avoid_x_location -= 1
                    self.center_index = self.avoid_x_location
                    angle = self.center_index - 640
                    speed = 15
                elif self.heading >= gt_heading:
                    self.start_avoid = False

        if self.gt_heading >=0:
            if current_lane == 3:
                if self.heading <= gt_heading + 18:
                    self.avoid_x_location -= 2
                    self.center_index = self.avoid_x_location
                    angle = self.center_index - 640
                    speed = 15

                elif self.heading > gt_heading + 18:
                    self.current_lane -= 1
                    self.avoid_x_location  = 640

                if self.heading > gt_heading:
                    self.avoid_x_location += 1
                    self.center_index = self.avoid_x_location
                    angle = self.center_index - 640
                    speed = 15
                elif self.heading <= gt_heading:
                    self.start_avoid = False
            
            else:
                if self.heading >= gt_heading - 18:
                    self.avoid_x_location += 2
                    self.center_index = self.avoid_x_location
                    angle = self.center_index - 640
                    speed = 15

                elif self.heading > gt_heading - 18:
                    self.current_lane += 1
                    self.avoid_x_location  = 640

                if self.heading < gt_heading:
                    self.avoid_x_location -= 1
                    self.center_index = self.avoid_x_location
                    angle = self.center_index - 640
                    speed = 15
                elif self.heading >= gt_heading:
                    self.start_avoid = False
        print(f'self.avoid_x_location: {self.avoid_x_location}')

        return angle, speed

    def is_something_front(self, obstacle_info):# [10.929349899291992, -0.095023512840271, -0.7778873443603516, 0.8449230194091797, 1.8444271087646484, 0.415688157081604]
        if len(obstacle_info)>=1:
            if 5 <= obstacle_info[0][0]< 20.0 and abs(obstacle_info[0][1]) < 1.0 and obstacle_info[0][4]> 1.0:
                return True
            else: return False
        else:
            return False
            
        
    

if __name__ == "__main__":
    try: 
        lane_detection_node = LaneDetection()
    except rospy.ROSInterruptException:
        pass
