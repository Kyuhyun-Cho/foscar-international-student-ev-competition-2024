#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import Path
from std_msgs.msg import Int64MultiArray, Float64, Int64, Bool
from sensor_msgs.msg import Imu
from geometry_msgs.msg import  Vector3
from visualization_msgs.msg import Marker, MarkerArray
from pyproj import Proj, transform
from morai_msgs.msg import GPSMessage, CtrlCmd, EventInfo
from morai_msgs.srv import MoraiEventCmdSrv
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from utils_sangam import pathReader,findLocalPath,purePursuit,rotateLiDAR2GPS, CCW
from lidar_object_detection.msg import ObjectInfo
from lidar_cam_fusion.msg import Float64Array2D
from ultralytics_ros.msg import YoloResult

import tf
from math import *
import numpy as np
from tabulate import tabulate
import os
import time 

# 아이오닉 5 -> 조향값(servo_msg) 0일 때 직진 양수이면 좌회전 음수이면 우회전

class EgoStatus:
    def __init__(self):
        self.position = Vector3()
        self.heading = 0.0
        self.velocity = Vector3()


class PurePursuit:
    def __init__(self):
        rospy.init_node('pure_pursuit', anonymous=True)

        self.path_name = 'sangam'

        # 속도 50, 30 구간을 나누기 위한 변수 
        self.speed_limit = 30


        # Lattice Planner Parameters
        lattice_path_length = 3
        lattice_current_lane = 2

        # Publisher
        self.global_path_pub                = rospy.Publisher('/global_path', Path, queue_size=1) ## global_path publisher 
        self.local_path_pub                 = rospy.Publisher('/local_path', Path, queue_size=1)
        self.heading_pub                    = rospy.Publisher('/heading', Float64, queue_size=1)
        self.current_waypoint_pub           = rospy.Publisher('/current_waypoint', Int64, queue_size=1)
        self.ctrl_cmd_pub                   = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)

        self.lattice_obstacle_pub           = rospy.Publisher('/lattice_obstacle_marker_array', MarkerArray, queue_size=1)
        self.acc_obstacle_pub               = rospy.Publisher('/acc_obstacle_marker_array', MarkerArray, queue_size=1)
        self.dynamic_obstacle_pub           = rospy.Publisher('/dynamic_obstacle_marker_array', MarkerArray, queue_size=1)

        self.pure_pursuit_target_point_pub  = rospy.Publisher('/pure_pusuit_target_point', Marker, queue_size=1)
        self.curvature_target_point_pub     = rospy.Publisher('/curvature_target_point', Marker, queue_size=1)
        self.ego_marker_pub                 = rospy.Publisher('/ego_marker', Marker, queue_size=1)
        
        ########################  lattice  ########################
        for i in range(1,lattice_path_length+1):            
            globals()['lattice_path_{}_pub'.format(i)]=rospy.Publisher('lattice_path_{}'.format(i),Path,queue_size=1)  

        for i in range(1,lattice_path_length+1):            
            globals()['planning_lattice_path_{}_pub'.format(i)]=rospy.Publisher('planning_lattice_path_{}'.format(i),Path,queue_size=1)  
        ########################  lattice  ########################

        # Subscriber
        rospy.Subscriber("/gps", GPSMessage, self.gpsCB) ## Vehicle Status Subscriber 
        rospy.Subscriber("/imu", Imu, self.imuCB) ## Vehicle Status Subscriber
        rospy.Subscriber("/traffic_light", Int64MultiArray, self.trafficlightCB)
        rospy.Subscriber("/object_info_lattice", ObjectInfo, self.latticeLidarObjectCB)
        rospy.Subscriber("/object_info_acc", ObjectInfo, self.accLidarObjectCB)
        rospy.Subscriber("/object_info_rotary", ObjectInfo, self.rotaryLidarObjectCB)
        rospy.Subscriber("/fusion_result", Float64Array2D, self.fusionResultCB)
        rospy.Subscriber("/lane_ctrl_cmd", CtrlCmd, self.laneCtrlCmdCB)
        rospy.Subscriber("/stopline_flag", Bool, self.stopLineCB)
        rospy.Subscriber("/yolo_result", YoloResult, self.yoloResultCB)
        

        self.status_msg   = EgoStatus()
        self.ctrl_cmd_msg = CtrlCmd()

        ### 동적 장애물 미션 파라미터 ###
        self.clock_wise = 0
        self.fusion_result_drum = []
        self.fusion_result_person = []
        self.is_dynamic_obstacle = False
        self.dynamic_obstacle_distance_threshold = 0.0
        self.min_distance = 99999
        self.min_path_coord = []
        self.min_obstacle_coord = []
        self.perception_not_working = 0 
        ###############################

        self.perception_really_not_working = False

        self.lattice_lidar_obstacle_info = []
        self.acc_lidar_obstacle_info = []
        self.rotary_lidar_obstacle_info = []

        self.lattice_obstacle_info = []
        self.acc_obstacle_info = []
        self.dynamic_obstacle_info = []

        self.yolo_bbox_size_list = []

        self.brake_cnt = 0

        self.rotary_stop_cnt = 0
        self.is_rotary_entered = False

        self.is_lab_time_check_started = False
        self.is_lab_time_check_finished = False

        self.is_status = False
        self.is_gps = False
        self.is_imu = False
        self.euler_data = [0,0,0,0]
        self.quaternion_data = [0,0,0,0]

        self.steering_angle_to_servo_offset = 0.0 ## servo moter offset
        self.target_x = 0.0
        self.target_y = 0.0
        self.curvature_target_x = 0.0
        self.curvature_target_y = 0.0
        self.corner_theta_degree = 0.0

        self.motor_msg = 0.0
        self.servo_msg = 0.0
        self.brake_msg = 0.0

        # -- 차선 커맨드 변수 -- #
        self.lane_ctrl_cmd_motor_msg = 0.0
        self.lane_ctrl_cmd_servo_msg = 0.0
        self.lane_ctrl_cmd_brake_msg = 0.0

        self.lane_cmd_2_motor_msg = 0.0
        self.lane_cmd_2_servo_msg = 0.0
        self.lane_cmd_2_brake_msg = 0.0
        # ---------------------- #

        self.steering_offset = 0.015
        # self.steering_offset = 0.05 

        self.lfd = 0.0
        self.min_distance_from_path = 0.0
        

        self.curve_servo_msg = 0.0
        self.curve_motor_msg = 0.0

        self.target_velocity_array = []

        self.is_dynamic_obstacle_cnt = 0

        ########traffic_stop_#######
        self.green_light_count = 0
        self.red_light_count = 0

        self.stopline_flag = False
        self.current_waypoint = 0
        self.mission_name = "Default"

        # path 별로 traiffic 위치 변경
 
        self.traffic_stop_index_1 = 1390
        self.traffic_stop_index_2 = 1847
        
        self.selected_lane = 1

        self.lattice_distance_threshold = 0.0
        self.acc_distance_threshold = 0.0
        self.dynamic_obstacle_distance_threshold = 0.0

        self.proj_UTM = Proj(proj='utm', zone = 52, elips='WGS84', preserve_units=False)
        self.tf_broadcaster = tf.TransformBroadcaster()

        ######################################## For Service ########################################
        rospy.wait_for_service('/Service_MoraiEventCmd')
        self.req_service = rospy.ServiceProxy('/Service_MoraiEventCmd', MoraiEventCmdSrv)
        self.req = EventInfo()

        self.forward_mode()
        #############################################################################################

        # Class
        path_reader = pathReader('path_maker') ## 경로 파일의 위치
        self.pure_pursuit = purePursuit() ## purePursuit import

        # Read path
        self.global_path = path_reader.read_txt(self.path_name+".txt") ## 출력할 경로의 이름

        rate = rospy.Rate(40) 
                                           
        while not rospy.is_shutdown():

            self.getEgoStatus()
            
            if self.is_status == True:
                # print(f'self.is_dynamic_obstacle_cnt: {self.is_dynamic_obstacle_cnt}')
                self.mission_name = "Default"
                # print("self.is_status: ", self.is_status)

                self.ctrl_cmd_msg.longlCmdType = 2

                local_path, self.current_waypoint, self.min_distance_from_path = findLocalPath(self.global_path, self.status_msg)

                if self.path_name == 'sangam_avoid_path':
                    self.current_waypoint += 2580
                elif self.path_name == 'sangam_avoid_path_2':
                    self.current_waypoint += 3436
                elif self.path_name == 'sangam_avoid_path_3':
                    self.current_waypoint += 3690

                # print("current waypoint:", self.current_waypoint)
                if self.current_waypoint < 1457:
                    self.speed_limit = 30
                else:
                    self.speed_limit = 50

                ######################## LiDAR 좌표계에서 Detect 된 장애물 GPS 좌표계로 변환하는 구간 ########################
                self.lattice_obstacle_info = rotateLiDAR2GPS(self.lattice_lidar_obstacle_info, self.status_msg)
                
                self.acc_lidar_obstacle_info.sort()
                self.acc_obstacle_info = rotateLiDAR2GPS(self.acc_lidar_obstacle_info, self.status_msg, self.current_waypoint)

                self.dynamic_obstacle_info = rotateLiDAR2GPS(self.fusion_result_person, self.status_msg, True)
                ###########################################################################################################
                #################### 동적 장애물 미션 ####################
                if (645 <= self.current_waypoint <= 1299) or (3380 <= self.current_waypoint <= 3700):
                    self.mission_name = "Dynamic"
                    # print("PERCEPTION NOT WORKING: ", self.perceqsssssssption_not_working)


                    # if (645 <= self.current_waypoint <= 1299):
                    #     self.speed_limit = 20
                    # elif  (3380 <= self.current_waypoint <= 3700):
                    #     self.speed_limit = 20
                    
                    self.speed_limit = 30

                    if any(size >= 480.0 for size in self.yolo_bbox_size_list):
                        # self.path_name = 'sangam.txt'
                        self.speed_limit = 10

                    if len(self.fusion_result_person) > 0:
                        self.min_distance, self.min_path_coord, self.min_obstacle_coord = self.pure_pursuit.getMinDistance(local_path, self.dynamic_obstacle_info, self.status_msg)
                        self.clock_wise = CCW(self.status_msg, self.min_path_coord, self.min_obstacle_coord)
                        self.is_dynamic_obstacle, self.dynamic_obstacle_distance_threshold = self.pure_pursuit.checkDynamicObstacle(self.clock_wise, self.min_distance, self.current_waypoint)

                        self.perception_not_working = 0
                    else:
                        self.perception_not_working += 1
                        self.is_dynamic_obstacle = False

                    if self.is_dynamic_obstacle is True:
                        self.is_dynamic_obstacle_cnt += 1
                    elif self.is_dynamic_obstacle is False:
                        self.is_dynamic_obstacle_cnt -= 25
                        if self.is_dynamic_obstacle_cnt < 0:
                            self.is_dynamic_obstacle_cnt = 0

                    if self.perception_not_working > 60:
                        ### 
                        # self.perception_really_not_working = True

                        self.perception_not_working = 0
                        self.pure_pursuit.is_obstacle_passed = False
                        self.pure_pursuit.first_clock_wise = None

                    if self.is_dynamic_obstacle is False:
                        self.brake_cnt = 0

                    if self.brake_cnt > 10000 and len(self.dynamic_obstacle_info) > 0:
                        self.motor_msg = 3
                        obstacle_dist = sqrt((self.dynamic_obstacle_info[0][0] - self.previous_obstacle_position[0][0])**2 + (self.dynamic_obstacle_info[0][1] - self.previous_obstacle_position[0][1])**2)
                        if obstacle_dist > 2.0:
                            self.brake_cnt = 0

                    elif self.is_dynamic_obstacle is True:
                        self.motor_msg = 0
                        self.brake_cnt += 1
                        self.previous_obstacle_position = self.dynamic_obstacle_info
                        self.brake()
                        continue

                    else:
                        self.motor_msg = 20
                #########################################################

                if 3701 <= self.is_dynamic_obstacle_cnt:
                    self.is_dynamic_obstacle_cnt = 0

                ########################  lattice  ########################
                # if 2579 <= self.current_waypoint <= 3054: #1500: # 3050
                #     self.mission_name = "Lattice"
                #     lattice_current_lane = 1
                #     self.speed_limit = 20
                #     lattice_path, planning_lattice_path, selected_lane, self.lattice_distance_threshold = self.pure_pursuit.latticePlanner(local_path, self.lattice_obstacle_info, self.status_msg, lattice_current_lane, 'first')
                
                #     lattice_current_lane = selected_lane
            
                #     if selected_lane != -1: 
                #         local_path = lattice_path[selected_lane]                
                    

                #     if len(lattice_path)==lattice_path_length:                    
                #         for i in range(1,lattice_path_length+1):
                #             globals()['lattice_path_{}_pub'.format(i)].publish(lattice_path[i-1])
                #             globals()['planning_lattice_path_{}_pub'.format(i)].publish(planning_lattice_path[i-1])


                # elif 3505 <= self.current_waypoint <= 3735:
                #     self.mission_name = "Lattice"
                #     lattice_current_lane = 2
                #     self.speed_limit = 20

                #     lattice_path, planning_lattice_path, selected_lane, self.lattice_distance_threshold = self.pure_pursuit.latticePlanner(local_path, self.lattice_obstacle_info, self.status_msg, lattice_current_lane, 'second')
                
                #     lattice_current_lane = selected_lane
            
                #     if selected_lane != -1: 
                #         local_path = lattice_path[selected_lane]                
                    

                #     if len(lattice_path)==lattice_path_length:                    
                #         for i in range(1,lattice_path_length+1):
                #             globals()['lattice_path_{}_pub'.format(i)].publish(lattice_path[i-1])
                #             globals()['planning_lattice_path_{}_pub'.format(i)].publish(planning_lattice_path[i-1])                    

                # if 3693 <= self.current_waypoint:
                #     self.mission_name = "Lattice"
                #     lattice_current_lane = 2
                #     self.speed_limit = 20

                #     lattice_path, planning_lattice_path, selected_lane, self.lattice_distance_threshold = self.pure_pursuit.latticePlanner(local_path, self.lattice_obstacle_info, self.status_msg, lattice_current_lane, 'third')
                
                #     lattice_current_lane = selected_lane
            
                #     if selected_lane != -1: 
                #         local_path = lattice_path[selected_lane]                
                    

                #     if len(lattice_path)==lattice_path_length:                    
                #         for i in range(1,lattice_path_length+1):
                #             globals()['lattice_path_{}_pub'.format(i)].publish(lattice_path[i-1])
                #             globals()['planning_lattice_path_{}_pub'.format(i)].publish(planning_lattice_path[i-1])


                #################################################################

                self.pure_pursuit.getPath(local_path) ## pure_pursuit 알고리즘에 Local path 적용
                self.pure_pursuit.getEgoStatus(self.status_msg) 

                is_obstacle_on_path, distance_object_to_car_list, self.acc_distance_threshold = self.pure_pursuit.isObstacleOnPath(local_path, self.acc_obstacle_info, self.status_msg, self.current_waypoint)

                self.steering, self.target_x, self.target_y, self.lfd = self.pure_pursuit.steeringAngle(0, self.current_waypoint)
                # print(f"self.steering: {self.steering}")
                # print(f'self.heading: {self.status_msg.heading}')
                self.corner_theta_degree, self.curvature_target_x, self.curvature_target_y = self.pure_pursuit.estimateCurvature()

                
                ####################################################


                
                ##################### 정적 장애물 - 패스 스위칭 #####################
                if 2580 <= self.current_waypoint <= 2982:
                    self.mission_name = "Switching"
                    static_lfd = 8
                    self.steering, self.target_x, self.target_y, self.lfd = self.pure_pursuit.steeringAngle(static_lfd)

                    is_obstacle_on_path, distance_object_to_car_list, self.lattice_distance_threshold = self.pure_pursuit.isObstacleOnPath(local_path, self.lattice_obstacle_info, self.status_msg, self.current_waypoint)

                    self.speed_limit = 40

                    if is_obstacle_on_path and (self.min_distance_from_path < 0.7) and (-33.0 <self.status_msg.heading < -24.0):

                        
                        if self.path_name == 'sangam' and self.isLeftEmpty():
                            self.path_name = 'sangam_avoid_path'
                            
                            self.global_path = path_reader.read_txt(self.path_name+".txt")
                            local_path, self.current_waypoint, self.min_distance_from_path = findLocalPath(self.global_path, self.status_msg)
                            self.pure_pursuit.getPath(local_path)
                            self.steering, self.target_x, self.target_y, self.lfd = self.pure_pursuit.steeringAngle(static_lfd)

                        elif self.path_name == 'sangam_avoid_path' and self.isRightEmpty():
                            self.path_name = 'sangam'

                            self.global_path = path_reader.read_txt(self.path_name+".txt")
                            local_path, self.current_waypoint, self.min_distance_from_path = findLocalPath(self.global_path, self.status_msg)
                            self.pure_pursuit.getPath(local_path)
                            self.steering, self.target_x, self.target_y, self.lfd = self.pure_pursuit.steeringAngle(static_lfd)

                    if self.path_name == 'sangam_avoid_path' and self.current_waypoint + 9 + 50 >= len(self.global_path.poses) + 2580: 
                        self.path_name = 'sangam'
                        self.global_path = path_reader.read_txt(self.path_name+".txt")


                elif 3436 <= self.current_waypoint <= 3664:
                    
                    if self.is_dynamic_obstacle_cnt == 0:
    
                        self.mission_name = "Switching"
                        static_lfd = 8
                        self.steering, self.target_x, self.target_y, self.lfd = self.pure_pursuit.steeringAngle(static_lfd, self.current_waypoint)

                        is_obstacle_on_path, distance_object_to_car_list, self.lattice_distance_threshold = self.pure_pursuit.isObstacleOnPath(local_path, self.lattice_obstacle_info, self.status_msg, self.current_waypoint)

                        self.speed_limit = 40

                        if any(size >= 480.0 for size in self.yolo_bbox_size_list):
                            # self.path_name = 'sangam.txt'
                            self.speed_limit = 10

                        if is_obstacle_on_path and (self.min_distance_from_path < 0.7) and (-134.0 <self.status_msg.heading < -124.0):
                            if self.path_name == 'sangam':
                                self.path_name = 'sangam_avoid_path_2'
                                
                            elif self.path_name == 'sangam_avoid_path_2':
                                self.path_name = 'sangam'

                            self.global_path = path_reader.read_txt(self.path_name+".txt")
                            local_path, self.current_waypoint, self.min_distance_from_path = findLocalPath(self.global_path, self.status_msg)
                            self.pure_pursuit.getPath(local_path)
                            self.steering, self.target_x, self.target_y, self.lfd = self.pure_pursuit.steeringAngle(static_lfd, self.current_waypoint)

                        if self.path_name == 'sangam_avoid_path_2' and self.current_waypoint + 9 + 28  >= len(self.global_path.poses) + 3436: 
                            self.path_name = 'sangam'
                            self.global_path = path_reader.read_txt(self.path_name+".txt")


                elif 3690 <= self.current_waypoint:
                    self.mission_name = "Switching"
                    static_lfd = 5
                    self.steering, self.target_x, self.target_y, self.lfd = self.pure_pursuit.steeringAngle(static_lfd, self.current_waypoint)

                    is_obstacle_on_path, distance_object_to_car_list, self.lattice_distance_threshold = self.pure_pursuit.isObstacleOnPath(local_path, self.lattice_obstacle_info, self.status_msg, self.current_waypoint)

                    self.speed_limit = 20

                    if is_obstacle_on_path and (self.min_distance_from_path < 0.7) and (-134.0 <self.status_msg.heading < -124.0):
                        
                        if self.path_name == 'sangam':
                            self.path_name = 'sangam_avoid_path_3'
                            
                        elif self.path_name == 'sangam_avoid_path_3':
                            self.path_name = 'sangam'
   
                        self.global_path = path_reader.read_txt(self.path_name+".txt")
                        local_path, self.current_waypoint, self.min_distance_from_path = findLocalPath(self.global_path, self.status_msg)
                        self.pure_pursuit.getPath(local_path)
                        self.steering, self.target_x, self.target_y, self.lfd = self.pure_pursuit.steeringAngle(static_lfd, self.current_waypoint)

                    # if self.path_name == 'sangam_avoid_path_2' and self.current_waypoint + 9 + 28  >= len(self.global_path.poses) + 3436: 
                    #     self.path_name = 'sangam'
                    #     self.global_path = path_reader.read_txt(self.path_name+".txt")
                


                # 
                # if self.path_name == 'sangam_avoid_path_2':
                #     self.speed_limit = 10

                ################################################################


                
                # 진입 전 감속
                if (1741 <= self.current_waypoint <= 1847):
                    self.mission_name = "신호등 전 감속"
                    self.speed_limit = 20
                elif (1935 <= self.current_waypoint <= 1986):
                    self.mission_name = "차선변경 감속"
                    self.speed_limit = 20
                elif (2983 <= self.current_waypoint <= 3050):
                    self.mission_name = "음영구간 전 감속"
                    self.speed_limit = 20
                elif (3662 <= self.current_waypoint <= 3693): # 3662 <= 3693
                    self.mission_name = "정적 전 감속"
                    self.speed_limit = 10
    



                if self.motor_msg != 3:
                    self.motor_msg = self.cornerController()
                self.servo_msg = (self.steering + 2.7) * self.steering_offset
                self.brake_msg = 0



                ######################## ACC 미션 #######################
                if self.current_waypoint <= 670 or 1457 <= self.current_waypoint < 2542:
                    self.mission_name = "ACC"

                    if is_obstacle_on_path:
                        # self.motor_msg = 5*(distance_object_to_car-3) - 35
                        # self.motor_msg = 0.3*(distance_object_to_car-10)**2

                        distance_object_to_car_list.sort()
                        distance_object_to_car = distance_object_to_car_list[0]
                        
                        self.motor_msg = 2.5 * distance_object_to_car - 25
                        if self.motor_msg <= 0: 
                            self.motor_msg = 0

                        elif self.motor_msg >= self.speed_limit-1: 
                            self.motor_msg = self.speed_limit-1 # 혹시 모를 내리막길이 있을 경우 발행 속도를 초과할 수 도 있기 때문에 안전하게 29, 49 까지만 발행하기


                        acc_y_list = [acc_info[1] for acc_info in self.acc_lidar_obstacle_info]

                        if self.motor_msg == 0:
                            if not any(abs(y) <= 2.0 for y in acc_y_list):
                                if (1741 <= self.current_waypoint <= 1795) or (1935 <= self.current_waypoint <= 1986): #차선변경구간
                                    self.mission_name = "차선 변경 구간"
                                    pass
                                else:
                                    self.motor_msg = 1
                #########################################################

                ###################### 로터리 미션 #######################
                if 1300 <= self.current_waypoint <= 1310:
                    self.mission_name = "Rotary"
                    if self.rotary_stop_cnt < 40:
                        self.rotary_stop_cnt += 1
                        self.motor_msg = 0

                    else:
                        if len(self.rotary_lidar_obstacle_info) > 0:
                            self.motor_msg = 0
                ########################################################


                if self.isRedLight() and self.isTrafficLightArea():
                    self.brake()
                    continue


                self.visualizeTargetPoint()
                self.visualizeCurvatureTargetPoint()
                self.visualizeEgoMarker()

                self.visualizeLatticeObstacle()
                self.visualizeAccObstacle()
                self.visualizeDynamicObstacle()
                
                
                self.local_path_pub.publish(local_path)
                self.global_path_pub.publish(self.global_path)
                self.heading_pub.publish(self.status_msg.heading)
                self.current_waypoint_pub.publish(self.current_waypoint)

                ####################### 종료 정지선 브레이크 #######################
                if self.current_waypoint >= 3919:
                    self.mission_name = "The End"
                    self.brake()
                    self.parking()
                    continue
                ################################################################

                ########################################################################################################################################################
                self.publishCtrlCmd(self.motor_msg, self.servo_msg, self.brake_msg)
                print("current_waypoint: ", self.current_waypoint)
                ########################################################################################################################################################

            else:
                self.mission_name = "GPS Blackout"
                self.heading_pub.publish(self.status_msg.heading)
                if self.is_gps is False:
                    
                    if self.traffic_stop_index_1-5 <= self.current_waypoint < 1450:
                        if self.isRedLight() and self.stopline_flag is True:
                            self.brake()
                            continue

                    self.motor_msg = self.lane_ctrl_cmd_motor_msg
                    self.servo_msg = self.lane_ctrl_cmd_servo_msg
                    self.brake_msg = self.lane_ctrl_cmd_brake_msg
            
                    self.publishCtrlCmd(self.motor_msg, self.servo_msg, self.brake_msg)
            ##########################################printlog#########################################
            # log = [
            #     {
            #         "Mission": self.mission_name, 
            #         "Waypoint": self.current_waypoint, 
            #         "Speed": self.motor_msg, 
            #         "Speed Limit": self.speed_limit, 
            #         "Steering": self.servo_msg,
            #         "LFD": self.lfd, 
            #     }
            # ]
            # os.system("clear")
            # print(tabulate(log, headers="keys", tablefmt="grid"))

            rate.sleep()
    
###################################################################### Service Request  ######################################################################
    # option - 1 : ctrl_mode / 2 : gear / 4 : lamps / 6 : gear + lamps
    # gear - 1: P / 2 : R / 3 : N / 4 : D
##############################################################################################################################################################

    def forward_mode(self):
        self.req.option = 6
        self.req.gear = 4
        self.req.lamps.turnSignal = 0
        self.req.lamps.emergencySignal = 0
        response = self.req_service(self.req)
        self.yaw_rear = False

    def rear_mode(self):
        self.req.option = 2
        self.req.gear = 2
        response = self.req_service(self.req)
        self.yaw_rear = True

    def drive_left_signal(self):
        self.req.option = 6
        self.req.gear = 4
        self.req.lamps.turnSignal = 1
        response = self.req_service(self.req)

    
    def drive_right_signal(self) :
        self.req.option = 6
        self.req.gear = 4
        self.req.lamps.turnSignal = 2
        response = self.req_service(self.req)

    def emergency_mode(self) :
        self.req.option = 6
        self.req.gear = 4
        self.req.lamps.emergencySignal = 1
        response = self.req_service(self.req)

    def parking(self) :
        self.req.option = 6
        self.req.gear = 1
        self.req.lamps.turnSignal = 0
        response = self.req_service(self.req)

    def brake(self) :
        self.ctrl_cmd_msg.longlCmdType = 2
        self.motor_msg = 0.0
        self.servo_msg = 0.0
        self.brake_msg = 1.0
        self.publishCtrlCmd(self.motor_msg, self.servo_msg, self.brake_msg)
    
###################################################################### Call Back ######################################################################
    def getEgoStatus(self): ## Vehicle Status Subscriber 
        if self.is_gps == True and self.is_imu == True:
            self.status_msg.position.x = self.xy_zone[0] - 313008.55819800857
            self.status_msg.position.y = self.xy_zone[1] - 4161698.628368007
            self.status_msg.position.z = 0.0
            self.status_msg.heading = self.euler_data[2] * 180/pi
            self.status_msg.velocity.x = self.motor_msg #self.velocity

            self.tf_broadcaster.sendTransform((self.status_msg.position.x, self.status_msg.position.y, self.status_msg.position.z),
                            tf.transformations.quaternion_from_euler(0, 0, (self.status_msg.heading)/180*pi),
                            rospy.Time.now(),
                            "base_link",
                            "map")
   
            self.is_status=True

        elif self.is_gps is False and self.is_imu is True:
            self.status_msg.heading = self.euler_data[2] * 180/pi
            self.is_status=False

        else:
            # print("Waiting for GPS & IMU")
            self.is_status=False


    def gpsCB(self, msg):
        if msg.status == 0: 
            # self.current_waypoint = -1
            self.is_gps = False

        else:
            self.xy_zone = self.proj_UTM(msg.longitude, msg.latitude)
            
            self.tf_broadcaster.sendTransform((0, 0, 1.18),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            "gps",
                            "base_link")
            
            self.tf_broadcaster.sendTransform((4.20, 0, 0.20),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            "velodyne",
                            "base_link")
            
            self.is_gps = True


    def imuCB(self, msg):
        self.quaternion_data = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        self.euler_data = tf.transformations.euler_from_quaternion((msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w))

        self.tf_broadcaster.sendTransform((-0.08, 0.0, 1.18),
                        tf.transformations.quaternion_from_euler(0, 0, 0),
                        rospy.Time.now(),
                        "imu",
                        "base_link")

        self.is_imu = True

    def trafficlightCB(self, msg):
        self.red_light_count   = msg.data[0]
        self.green_light_count = msg.data[1]
        # print("Green:", self.green_light_count, 'Red:', self.red_light_count)

    def latticeLidarObjectCB(self, msg):
        self.lattice_lidar_obstacle_info = [[] for i in range(msg.objectCounts)]
        for i in range(msg.objectCounts):
            self.lattice_lidar_obstacle_info[i] = [msg.centerX[i], msg.centerY[i], msg.centerZ[i], msg.lengthX[i], msg.lengthY[i], msg.lengthZ[i]]

    def accLidarObjectCB(self, msg):
        self.acc_lidar_obstacle_info = [[] for i in range(msg.objectCounts)]
        for i in range(msg.objectCounts):
            self.acc_lidar_obstacle_info[i] = [msg.centerX[i], msg.centerY[i], msg.centerZ[i], msg.lengthX[i], msg.lengthY[i], msg.lengthZ[i]]

    def rotaryLidarObjectCB(self, msg):
        self.rotary_lidar_obstacle_info = [[] for i in range(msg.objectCounts)]
        for i in range(msg.objectCounts):
            self.rotary_lidar_obstacle_info[i] = [msg.centerX[i], msg.centerY[i], msg.centerZ[i], msg.lengthX[i], msg.lengthY[i], msg.lengthZ[i]]
    
    def fusionResultCB(self, msg):
        self.fusion_result_drum = [list(bbox.bbox)[4:8] for bbox in msg.bboxes if bbox.bbox[7] == 0]
        self.fusion_result_person = [list(bbox.bbox)[4:8] for bbox in msg.bboxes if bbox.bbox[7] == 1]

    def laneCtrlCmdCB(self, msg):
        self.lane_ctrl_cmd_motor_msg = msg.velocity
        self.lane_ctrl_cmd_servo_msg = msg.steering
        self.lane_ctrl_cmd_brake_msg = msg.brake

    def stopLineCB(self, msg):
        self.stopline_flag = msg.data
        # print(self.stopline_flag)

    def yoloResultCB(self, msg):
        detections_list = msg.detections.detections
        self.yolo_bbox_size_list = [0.0 for i in range(len(detections_list))]

        for i in range(len(detections_list)):
            self.yolo_bbox_size_list[i] = detections_list[i].bbox.size_x * detections_list[i].bbox.size_y
            
    




###################################################################### Function ######################################################################

    def publishCtrlCmd(self, motor_msg, servo_msg, brake_msg):
        self.ctrl_cmd_msg.velocity = motor_msg
        self.ctrl_cmd_msg.steering = servo_msg
        self.ctrl_cmd_msg.brake = brake_msg
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

    def setBrakeMsgWithNum(self, brake):
        self.brake_msg = brake
    
    def cornerController(self):
        if self.corner_theta_degree > 50:
            self.corner_theta_degree = 50

        # motor_msg = 150/(self.corner_theta_degree+4) + 12.5
        motor_msg = 20
        # 속도 제한 구역에 따른 그래프 - 1차 함수로 테스트 해보고있음(준호)
        # 혹시 모를 내리막길이 있을 경우 발행 속도를 초과할 수 도 있기 때문에 안전하게 29, 49 까지만 발행하기
        if self.speed_limit == 50:
            motor_msg = -0.7 * self.corner_theta_degree + 49
        elif self.speed_limit == 40:
            motor_msg = -0.52 * self.corner_theta_degree + 40
        elif self.speed_limit == 30:
            motor_msg = -0.3 * self.corner_theta_degree + 20
        else:
            motor_msg = self.speed_limit
        # print(f"motor_msg: {motor_msg}")

        if motor_msg < 0:
            motor_msg = 5

        return motor_msg 

    def visualizeLatticeObstacle(self):
        lattice_obstacle_array = MarkerArray()

        for i in range(len(self.lattice_obstacle_info)):
            lattice_obstacle = Marker()
            lattice_obstacle.header.frame_id = "map"
            lattice_obstacle.id = i
            lattice_obstacle.type = lattice_obstacle.CYLINDER
            lattice_obstacle.action = lattice_obstacle.ADD
            lattice_obstacle.scale.x = self.lattice_distance_threshold * 2
            lattice_obstacle.scale.y = self.lattice_distance_threshold * 2
            lattice_obstacle.scale.z = 2.0
            lattice_obstacle.pose.orientation.w = 1.0
            lattice_obstacle.color.r = 1.0
            lattice_obstacle.color.g = 1.0
            lattice_obstacle.color.b = 1.0
            lattice_obstacle.color.a = 0.5 
            lattice_obstacle.pose.position.x = self.lattice_obstacle_info[i][0]
            lattice_obstacle.pose.position.y = self.lattice_obstacle_info[i][1]
            lattice_obstacle.pose.position.z = 0.0
            lattice_obstacle.lifetime = rospy.Duration(0.1)

            lattice_obstacle_array.markers.append(lattice_obstacle)

        self.lattice_obstacle_pub.publish(lattice_obstacle_array)

    def visualizeDynamicObstacle(self):
        dynamic_obstacle_array = MarkerArray()

        for i in range(len(self.dynamic_obstacle_info)):
            dynamic_obstacle = Marker()
            dynamic_obstacle.header.frame_id = "map"
            dynamic_obstacle.id = i
            dynamic_obstacle.type = dynamic_obstacle.CYLINDER
            dynamic_obstacle.action = dynamic_obstacle.ADD
            dynamic_obstacle.scale.x = self.dynamic_obstacle_distance_threshold * 2
            dynamic_obstacle.scale.y = self.dynamic_obstacle_distance_threshold * 2
            dynamic_obstacle.scale.z = 2.0
            dynamic_obstacle.pose.orientation.w = 1.0
            dynamic_obstacle.color.r = 0.0
            dynamic_obstacle.color.g = 1.0
            dynamic_obstacle.color.b = 0.0
            dynamic_obstacle.color.a = 0.5 
            dynamic_obstacle.pose.position.x = self.dynamic_obstacle_info[i][0]
            dynamic_obstacle.pose.position.y = self.dynamic_obstacle_info[i][1]
            dynamic_obstacle.pose.position.z = 0.0
            dynamic_obstacle.lifetime = rospy.Duration(0.1)

            dynamic_obstacle_array.markers.append(dynamic_obstacle)

        self.dynamic_obstacle_pub.publish(dynamic_obstacle_array)

    def visualizeAccObstacle(self):
        acc_obstacle_array = MarkerArray()

        for i in range(len(self.acc_obstacle_info)):
            acc_obstacle = Marker()
            acc_obstacle.header.frame_id = "map"
            acc_obstacle.type = acc_obstacle.CYLINDER
            acc_obstacle.action = acc_obstacle.ADD
            acc_obstacle.scale.x = self.acc_distance_threshold * 2
            acc_obstacle.scale.y = self.acc_distance_threshold * 2
            acc_obstacle.scale.z = 2.0
            acc_obstacle.pose.orientation.w = 1.0
            acc_obstacle.color.r = 1.0
            acc_obstacle.color.g = 0.0
            acc_obstacle.color.b = 0.0
            acc_obstacle.color.a = 0.5 
            acc_obstacle.pose.position.x = self.acc_obstacle_info[i][0]
            acc_obstacle.pose.position.y = self.acc_obstacle_info[i][1]
            acc_obstacle.pose.position.z = 0.0
            acc_obstacle.lifetime = rospy.Duration(0.1)

            acc_obstacle_array.markers.append(acc_obstacle)

        self.acc_obstacle_pub.publish(acc_obstacle_array)


    def visualizeTargetPoint(self):
        pure_pursuit_target_point = Marker()
        pure_pursuit_target_point.header.frame_id = "map"
        pure_pursuit_target_point.type = pure_pursuit_target_point.SPHERE
        pure_pursuit_target_point.action = pure_pursuit_target_point.ADD
        pure_pursuit_target_point.scale.x = 1.0
        pure_pursuit_target_point.scale.y = 1.0
        pure_pursuit_target_point.scale.z = 1.0
        pure_pursuit_target_point.pose.orientation.w = 1.0
        pure_pursuit_target_point.color.r = 1.0
        pure_pursuit_target_point.color.g = 0.0
        pure_pursuit_target_point.color.b = 0.0
        pure_pursuit_target_point.color.a = 1.0 
        pure_pursuit_target_point.pose.position.x = self.target_x
        pure_pursuit_target_point.pose.position.y = self.target_y
        pure_pursuit_target_point.pose.position.z = 0.0
        
        self.pure_pursuit_target_point_pub.publish(pure_pursuit_target_point)


    def visualizeCurvatureTargetPoint(self):
        curvature_target_point = Marker()
        curvature_target_point.header.frame_id = "map"
        curvature_target_point.type = curvature_target_point.SPHERE
        curvature_target_point.action = curvature_target_point.ADD
        curvature_target_point.scale.x = 1.0
        curvature_target_point.scale.y = 1.0
        curvature_target_point.scale.z = 1.0
        curvature_target_point.pose.orientation.w = 1.0
        curvature_target_point.color.r = 0.0
        curvature_target_point.color.g = 0.0
        curvature_target_point.color.b = 1.0
        curvature_target_point.color.a = 1.0 
        curvature_target_point.pose.position.x = self.curvature_target_x
        curvature_target_point.pose.position.y = self.curvature_target_y
        curvature_target_point.pose.position.z = 0.0
        
        self.curvature_target_point_pub.publish(curvature_target_point)


    def visualizeEgoMarker(self):
        ego_marker = Marker()
        ego_marker.header.frame_id = "map"
        ego_marker.type = ego_marker.MESH_RESOURCE
        ego_marker.mesh_resource = "package://pure_pursuit/stl/egolf.stl"
        ego_marker.mesh_use_embedded_materials = True
        ego_marker.action = ego_marker.ADD
        ego_marker.scale.x = 1.2
        ego_marker.scale.y = 1.2
        ego_marker.scale.z = 1.2
        ego_marker.pose.orientation.x = self.quaternion_data[0]
        ego_marker.pose.orientation.y = self.quaternion_data[1]
        ego_marker.pose.orientation.z = self.quaternion_data[2]
        ego_marker.pose.orientation.w = self.quaternion_data[3]
        ego_marker.color.r = 1.0
        ego_marker.color.g = 1.0
        ego_marker.color.b = 1.0
        ego_marker.color.a = 1.0
        ego_marker.pose.position.x = self.status_msg.position.x
        ego_marker.pose.position.y = self.status_msg.position.y
        ego_marker.pose.position.z = 0.0
        
        self.ego_marker_pub.publish(ego_marker)


    def isRedLight(self):
        if self.green_light_count >= 25 and self.red_light_count <= 5:
            return False  # Green
        else:  
            return True   # Red

    def isTrafficLightArea(self):
        if ((self.traffic_stop_index_2 - self.motor_msg/3 <= self.current_waypoint <= self.traffic_stop_index_2)):#(self.current_waypoint <= self.traffic_stop_index_1+10) or 
            return True
        else:
            return False
        
    def isLeftEmpty(self):
        for i in range(len(self.lattice_lidar_obstacle_info)):
            x_min = self.lattice_lidar_obstacle_info[i][0] - self.lattice_lidar_obstacle_info[i][3]/2 
            x_max = self.lattice_lidar_obstacle_info[i][0] + self.lattice_lidar_obstacle_info[i][3]/2
            y_min = self.lattice_lidar_obstacle_info[i][1] - self.lattice_lidar_obstacle_info[i][4]/2
            # print("x_min: ", x_min)
            # print("x_max: ", x_max)
            # print("y_min: ", y_min)
            # if (x_min >= -1.5) and (2.0 <= y_min <= 6.5):

            if (x_max >= -1.5) and (2.0 <= y_min <= 6.5):
                return False
        return True
        

    def isRightEmpty(self):
        for i in range(len(self.lattice_lidar_obstacle_info)):
            x_min = self.lattice_lidar_obstacle_info[i][0] - self.lattice_lidar_obstacle_info[i][3]/2 
            x_max = self.lattice_lidar_obstacle_info[i][0] + self.lattice_lidar_obstacle_info[i][3]/2
            y_min = self.lattice_lidar_obstacle_info[i][1] - self.lattice_lidar_obstacle_info[i][4]/2
            y_max = self.lattice_lidar_obstacle_info[i][1] + self.lattice_lidar_obstacle_info[i][4]/2
            # if (x_min >= -1.5) and (-6.5 <= y_min <= -2.0):
            if (x_max >= -1.5) and (-6.5 <= y_max <= -2.0):
                return False
        return True
    

    def isLeftEmptyForDrum(self):
        for i in range(len(self.lattice_lidar_obstacle_info)):
            x_min = self.lattice_lidar_obstacle_info[i][0] - self.lattice_lidar_obstacle_info[i][3]/2 
            x_max = self.lattice_lidar_obstacle_info[i][0] + self.lattice_lidar_obstacle_info[i][3]/2
            y = self.lattice_lidar_obstacle_info[i][1] 
            # print("x_min: ", x_min)
            # print("x_max: ", x_max)
            # print("y_min: ", y_min)
            if (-2.0 <= x_max <= 5.0) and (2.0 <= y <= 4.0):
                return False
        return True
        

    def isRightEmptyForDrum(self):
        for i in range(len(self.lattice_lidar_obstacle_info)):
            x_min = self.lattice_lidar_obstacle_info[i][0] - self.lattice_lidar_obstacle_info[i][3]/2 
            x_max = self.lattice_lidar_obstacle_info[i][0] + self.lattice_lidar_obstacle_info[i][3]/2
            y_min = self.lattice_lidar_obstacle_info[i][1] - self.lattice_lidar_obstacle_info[i][4]/2
            y = self.lattice_lidar_obstacle_info[i][1]
            if (-2.0 <= x_max <= 5.0) and (-4.0 <= y <= -2.0):
                return False
        return True
    



if __name__ == '__main__':
    try:
        pure_pursuit_= PurePursuit()
    except rospy.ROSInterruptException:
        pass