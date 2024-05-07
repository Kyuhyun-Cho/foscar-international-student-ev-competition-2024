#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import Path
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import Marker, MarkerArray
from pyproj import Proj, transform
from morai_msgs.msg import GPSMessage, CtrlCmd, EventInfo
from morai_msgs.srv import MoraiEventCmdSrv
from lidar_cam_fusion.msg import Float64Array2D
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from utils_wonju import pathReader, findLocalPath, purePursuit, rotateLiDAR2GPS, CCW
from ultralytics_ros.msg import YoloResult
from lidar_object_detection.msg import ObjectInfo

import tf
import time
from math import *
import numpy as np

# 아이오닉 5 -> 조향값(servo_msg) 0일 때 직진 양수이면 좌회전 음수이면 우회전

class EgoStatus:
    def __init__(self):
        self.position = Vector3()
        self.heading = 0.0
        self.velocity = Vector3()


class PurePursuit:
    def __init__(self):
        rospy.init_node('pure_pursuit', anonymous=True)

        self.path_name = 'first'

        # Publisher
        self.global_path_pub  = rospy.Publisher('/global_path', Path, queue_size=1) ## global_path publisher 
        self.local_path_pub   = rospy.Publisher('/local_path', Path, queue_size=1)
        self.ctrl_cmd_pub     = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)

        self.fusion_obstacle_pub = rospy.Publisher('/fusion_obstacle_marker_array', MarkerArray, queue_size=1)
        self.lidar_obstacle_pub = rospy.Publisher('/lidar_obstacle_marker_array', MarkerArray, queue_size=1)
        self.fusion_obstacle_threshold_pub = rospy.Publisher('/fusion_obstacle_threshold_marker_array', MarkerArray, queue_size=1)
        self.lidar_obstacle_big_threshold_pub = rospy.Publisher('/lidar_obstacle_big_threshold_marker_array', MarkerArray, queue_size=1)
        self.lidar_obstacle_small_threshold_pub = rospy.Publisher('/lidar_obstacle_small_threshold_marker_array', MarkerArray, queue_size=1)
        self.pure_pursuit_target_point_pub = rospy.Publisher('/pure_pusuit_target_point', Marker, queue_size=1)
        self.curvature_target_point_pub = rospy.Publisher('/curvature_target_point', Marker, queue_size=1)
        self.ego_marker_pub = rospy.Publisher('/ego_marker', Marker, queue_size=1)

        # Subscriber
        rospy.Subscriber("/gps", GPSMessage, self.gpsCB) ## Vehicle Status Subscriber 
        rospy.Subscriber("/imu", Imu, self.imuCB) ## Vehicle Status Subscriber
        rospy.Subscriber("/fusion_result", Float64Array2D, self.fusionResultCB)
        # rospy.Subscriber("/bounding_box", MarkerArray, self.boundingboxCB)
        rospy.Subscriber("/object_info", ObjectInfo, self.boundingboxCB)

        rospy.Subscriber("/yolo_result", YoloResult, self.yolo_result_callback)

        self.status_msg   = EgoStatus()
        self.ctrl_cmd_msg = CtrlCmd()

        self.clock_wise = 0
        self.fusion_result = []
        self.lidar_result = []
        self.obstacle_info = []
        self.lidar_obstacle_info = []
        self.is_dynamic_obstacle = False
        self.distance_threshold = 0.0
        self.distance_threshold_small = 0.0
        self.distance_threshold_big = 0.0
        self.min_distance = 99999
        self.min_path_coord = []
        self.min_obstacle_coord = []
        self.perception_not_working = 0 
        self.perception_not_working_while_braking = 0

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

        self.steering_offset = 0.015 

        self.curve_servo_msg = 0.0
        self.curve_motor_msg = 0.0

        self.brake_cnt = 0
        self.previous_obstacle_position = 0.0

        self.target_velocity_array = []

        self.center_x_list = []
        self.center_y_list = []
        self.size_x_list = []
        self.size_y_list = []
        self.id_list = []

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
        self.global_path, self.target_velocity_array = path_reader.read_txt(self.path_name+".txt") ## 출력할 경로의 이름

        rate = rospy.Rate(40) 
                                           
        while not rospy.is_shutdown():
            
            self.getEgoStatus()
            
            if self.is_status == True:

                self.ctrl_cmd_msg.longlCmdType = 2

                local_path, current_waypoint = findLocalPath(self.global_path, self.status_msg)
                # if current_waypoint == 23 and self.is_lab_time_check_started == False:
                #     self.lab_start_time = time.time()
                #     self.is_lab_time_check_started = True
                #     print("######### Lab Time Check Start #########")
                
                print("Current Waypoint:", current_waypoint)

                # print("self.distatnce_threshold",self.distance_threshold)

                self.pure_pursuit.getPath(local_path) ## pure_pursuit 알고리즘에 Local path 적용
                self.pure_pursuit.getEgoStatus(self.status_msg) 

                # s 자 구간일 경우 라이다만으로
                if self.path_name == 'first' and current_waypoint >= 590:  # 680  #625
                    self.lidar_obstacle_info = rotateLiDAR2GPS(self.lidar_result, self.status_msg)

                self.obstacle_info = rotateLiDAR2GPS(self.fusion_result, self.status_msg)
                
                # print("self.lidar_obstacle_info: ", self.lidar_obstacle_info)
                # print("self.obstacle_info: ", self.obstacle_info)

                if len(self.fusion_result) > 0:
                    self.min_distance, self.min_path_coord, self.min_obstacle_coord = self.pure_pursuit.getMinDistance(local_path, self.obstacle_info, self.status_msg)
                    self.clock_wise = CCW(self.status_msg, self.min_path_coord, self.min_obstacle_coord)
                    self.is_dynamic_obstacle, self.is_slowdown_obstacle, self.distance_threshold, _ = self.pure_pursuit.checkDynamicObstacle(self.clock_wise, self.min_distance)

                    self.perception_not_working = 0
                else:
                    self.perception_not_working += 1
                    
                    if self.path_name == 'first' and current_waypoint >= 590 and len(self.lidar_obstacle_info) > 0:
                        self.min_distance, self.min_path_coord, self.min_obstacle_coord = self.pure_pursuit.getMinDistance(local_path, self.lidar_obstacle_info, self.status_msg)
                        self.is_dynamic_obstacle, self.is_slowdown_obstacle, self.distance_threshold_small, self.distance_threshold_big = self.pure_pursuit.checkDynamicObstacleInScurve(self.min_distance)
                        
                        # self.perception_not_working = 0
                            
                    else:
                        self.is_dynamic_obstacle = False
                        self.is_slowdown_obstacle = False
                    # print(self.perception_not_working)

                if self.perception_not_working > 80: #80
                    self.perception_not_working = 0
                    self.pure_pursuit.is_obstacle_passed = False
                    self.pure_pursuit.first_clock_wise = None
   
                self.visualizeFusionObstacle()


                # S자 구간에서만 예외처리
                if self.path_name == 'first' and current_waypoint >= 590:    #625
                    self.steering, self.target_x, self.target_y = self.pure_pursuit.steering_angle(1.5)
                    self.corner_theta_degree, self.curvature_target_x, self.curvature_target_y = self.pure_pursuit.corner_estimation()
                    self.motor_msg = 15
                    self.visualizeLidarObstacle()
                    # self.motor_msg = 3
                    # self.distance_threshold =1.5
                
                # 오르막 구간 예외처리
                # elif self.path_name == 'first' and 136 <= current_waypoint <= 186:    #625
                #     self.steering, self.target_x, self.target_y = self.pure_pursuit.steering_angle()
                #     self.corner_theta_degree, self.curvature_target_x, self.curvature_target_y = self.pure_pursuit.corner_estimation()
                #     self.motor_msg = 15
                    
                # 보편적인 상황
                else:
                    self.steering, self.target_x, self.target_y = self.pure_pursuit.steering_angle()
                    self.corner_theta_degree, self.curvature_target_x, self.curvature_target_y = self.pure_pursuit.corner_estimation()
                    self.motor_msg = self.corner_controller()

                self.servo_msg = self.steering * self.steering_offset
                self.brake_msg = 0

                self.visualizeTargetPoint()
                self.visualizeCurvatureTargetPoint()
                self.visualizeEgoMarker()

                if self.path_name == 'third' and current_waypoint >= 235:
                    self.brake()
                    self.parking()
                    # if self.is_lab_time_check_finished == False:
                    #     time_in_seconds = time.time() - self.lab_start_time
                    #     minutes, seconds = divmod(int(time_in_seconds), 60)
                    #     time_mm_ss = "{:02d}:{:02d}".format(minutes, seconds)
                    #     print("##### Lab Time:", time_mm_ss, "#####")
                    #     self.is_lab_time_check_finished = True
                    continue

                if self.path_name == 'first' and current_waypoint + 15 >= len(self.global_path.poses): 
                    self.path_name = 'second'
                    self.global_path, self.target_velocity_array = path_reader.read_txt(self.path_name+".txt")

                elif self.path_name == 'second' and current_waypoint + 25 >= len(self.global_path.poses): 
                    self.path_name = 'third'
                    self.global_path, self.target_velocity_array = path_reader.read_txt(self.path_name+".txt")

                self.local_path_pub.publish(local_path)
                self.global_path_pub.publish(self.global_path)

                ########################################################################################################################################################
                
                # brake cnt가 계속 쌓이는 문제 해결 위함.
                # if self.is_dynamic_obstacle is False:
                if self.perception_not_working_while_braking > 100:
                    self.brake_cnt = 0
                    self.perception_not_working_while_braking = 0


                if len(self.size_y_list) > 0 and ((1.4 * 90) /self.size_y_list[0]) * 10 <= 13:
                    # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!{((1.4 * 90) /self.size_y_list[0])*10}')
                    self.is_slowdown_obstacle_camera = True
                else:
                    self.is_slowdown_obstacle_camera = False

                # print("Obastacle 개수: ", len(self.obstacle_info))
                # print("중간에 인지 x: ", self.perception_not_working_while_braking)

                if self.brake_cnt > 300:
                    # print("!!!!")
                    self.publishCtrlCmd(3, self.servo_msg, self.brake_msg)
                    if len(self.obstacle_info) > 0:
                        self.perception_not_working_while_braking = 0
                        obstacle_dist = sqrt((self.obstacle_info[0][0] - self.previous_obstacle_position[0][0])**2 + (self.obstacle_info[0][1] - self.previous_obstacle_position[0][1])**2)
                        if obstacle_dist > 1.0: # 2.0
                            self.brake_cnt = 0
                    else:
                        self.perception_not_working_while_braking += 1
                        self.obstacle_info = [self.previous_obstacle_position[0][0], self.previous_obstacle_position[0][1]]

                elif self.is_dynamic_obstacle is True:
                    self.publishCtrlCmd(0.0, self.servo_msg, 0.0)
                    self.brake_cnt += 1

                    if len(self.obstacle_info) > 0:
                        self.previous_obstacle_position = self.obstacle_info
                    else:
                        self.previous_obstacle_position = self.lidar_obstacle_info
                
                elif (self.is_slowdown_obstacle is True) or (self.is_slowdown_obstacle_camera is True):
                    self.publishCtrlCmd(8.0, self.servo_msg, 0.0)
                    # self.brake_cnt += 1

                    if len(self.obstacle_info) > 0:
                        self.previous_obstacle_position = self.obstacle_info
                    elif len(self.lidar_obstacle_info) > 0:
                        self.previous_obstacle_position = self.lidar_obstacle_info
                    else:
                        pass
                    

                else:
                    self.publishCtrlCmd(self.motor_msg, self.servo_msg, self.brake_msg)
                    # self.publishCtrlCmd(18, self.servo_msg, self.brake_msg)

                # print(self.brake_cnt)
                ########################################################################################################################################################
            else:
                # print("Waiting for Status Msg")
                continue

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
            self.status_msg.position.x = self.xy_zone[0] - 402300.0
            self.status_msg.position.y = self.xy_zone[1] - 4132900.0
            self.status_msg.position.z = 0.0
            self.status_msg.heading = self.euler_data[2] * 180/pi
            self.status_msg.velocity.x = self.motor_msg #self.velocity

            self.tf_broadcaster.sendTransform((self.status_msg.position.x, self.status_msg.position.y, self.status_msg.position.z),
                            tf.transformations.quaternion_from_euler(0, 0, (self.status_msg.heading)/180*pi),
                            rospy.Time.now(),
                            "base_link",
                            "map")
   
            self.is_status=True
        else:
            print("Waiting for GPS & IMU")


    def gpsCB(self, msg):
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

    def fusionResultCB(self, msg):
        self.fusion_result = [list(bbox.bbox)[4:8] for bbox in msg.bboxes]

    def boundingboxCB(self, msg):
        # self.lidar_result = [[marker.pose.position.x, marker.pose.position.y, marker.pose.position.z, 0.0] for marker in msg.markers]

        self.lidar_result = [[] for i in range(msg.objectCounts)]
        for i in range(msg.objectCounts):
            self.lidar_result[i] = [msg.centerX[i], msg.centerY[i], msg.centerZ[i]]


    def yolo_result_callback(self, msg):
        bbox_list = [detection.bbox for detection in msg.detections.detections]

        self.size_y_list = [bbox.size_y for bbox in bbox_list]

        # self.size_x_list = [bbox.size_x for bbox in bbox_list]

        # result_list = [detection.results for detection in msg.detections.detections]

        # self.center_x_list = [bbox.center.x for bbox in bbox_list]
        # self.center_y_list = [bbox.center.y for bbox in bbox_list]


        # self.id_list = [result[0].id for result in result_list]
    



###################################################################### Function ######################################################################

    def publishCtrlCmd(self, motor_msg, servo_msg, brake_msg):
        self.ctrl_cmd_msg.velocity = motor_msg
        self.ctrl_cmd_msg.steering = servo_msg
        self.ctrl_cmd_msg.brake = brake_msg
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

    def setBrakeMsgWithNum(self, brake):
        self.brake_msg = brake
    
    def corner_controller(self):
        if self.corner_theta_degree > 50:
            self.corner_theta_degree = 50
        # motor_msg = 150/(self.corner_theta_degree + 3) + 13  # fast version = max 55km/h
        motor_msg = 150/(self.corner_theta_degree+8) + 15 # 13   safety_version
    

        return motor_msg 

######################################################### Visualize #########################################################

    def visualizeFusionObstacle(self):
        fusion_obstacle_array = MarkerArray()
        fusion_obstacle_threshold_array = MarkerArray()

        for i in range(len(self.obstacle_info)):
            fusion_obstacle = Marker()
            fusion_obstacle.header.frame_id = "map"
            fusion_obstacle.id = i
            fusion_obstacle.type = fusion_obstacle.MESH_RESOURCE
            fusion_obstacle.mesh_resource = "package://pure_pursuit/stl/person.stl"
            fusion_obstacle.mesh_use_embedded_materials = True
            fusion_obstacle.action = fusion_obstacle.ADD
            fusion_obstacle.scale.x = 0.05 
            fusion_obstacle.scale.y = 0.05 
            fusion_obstacle.scale.z = 0.05 
            fusion_obstacle.pose.orientation.w = 1.0
            fusion_obstacle.color.r = 0.0
            fusion_obstacle.color.g = 0.0
            fusion_obstacle.color.b = 1.0
            fusion_obstacle.color.a = 1.0
            fusion_obstacle.pose.position.x = self.obstacle_info[i][0] - 0.0
            fusion_obstacle.pose.position.y = self.obstacle_info[i][1] - 4.0
            fusion_obstacle.pose.position.z = 0.0 - 1.0
            fusion_obstacle.lifetime = rospy.Duration(0.1)

            fusion_obstacle_array.markers.append(fusion_obstacle)

            fusion_obstacle_threshold = Marker()
            fusion_obstacle_threshold.header.frame_id = "map"
            fusion_obstacle_threshold.id = i+100
            fusion_obstacle_threshold.type = fusion_obstacle_threshold.CYLINDER
            fusion_obstacle_threshold.action = fusion_obstacle.ADD
            fusion_obstacle_threshold.scale.x = self.distance_threshold * 2
            fusion_obstacle_threshold.scale.y = self.distance_threshold * 2 
            fusion_obstacle_threshold.scale.z = 0.2
            fusion_obstacle_threshold.pose.orientation.w = 1.0
            fusion_obstacle_threshold.color.r = 1.0
            fusion_obstacle_threshold.color.g = 1.0
            fusion_obstacle_threshold.color.b = 1.0
            fusion_obstacle_threshold.color.a = 0.2
            fusion_obstacle_threshold.pose.position.x = self.obstacle_info[i][0]
            fusion_obstacle_threshold.pose.position.y = self.obstacle_info[i][1]
            fusion_obstacle_threshold.pose.position.z = 0.0
            fusion_obstacle_threshold.lifetime = rospy.Duration(0.1)

            fusion_obstacle_threshold_array.markers.append(fusion_obstacle_threshold)


        self.fusion_obstacle_pub.publish(fusion_obstacle_array)
        self.fusion_obstacle_threshold_pub.publish(fusion_obstacle_threshold_array)

    def visualizeLidarObstacle(self):
        lidar_obstacle_array = MarkerArray()
        lidar_obstacle_big_threshold_array = MarkerArray()
        lidar_obstacle_small_threshold_array = MarkerArray()

        for i in range(len(self.lidar_obstacle_info)):
            lidar_obstacle = Marker()
            lidar_obstacle.header.frame_id = "map"
            lidar_obstacle.id = i
            lidar_obstacle.type = lidar_obstacle.MESH_RESOURCE
            lidar_obstacle.mesh_resource = "package://pure_pursuit/stl/person.stl"
            lidar_obstacle.mesh_use_embedded_materials = True
            lidar_obstacle.action = lidar_obstacle.ADD
            lidar_obstacle.scale.x = 0.05 
            lidar_obstacle.scale.y = 0.05 
            lidar_obstacle.scale.z = 0.05 
            lidar_obstacle.pose.orientation.w = 1.0
            lidar_obstacle.color.r = 0.0
            lidar_obstacle.color.g = 0.0
            lidar_obstacle.color.b = 1.0
            lidar_obstacle.color.a = 1.0
            lidar_obstacle.pose.position.x = self.lidar_obstacle_info[i][0] - 0.0
            lidar_obstacle.pose.position.y = self.lidar_obstacle_info[i][1] - 4.0
            lidar_obstacle.pose.position.z = 0.0 - 1.0
            lidar_obstacle.lifetime = rospy.Duration(0.1)

            lidar_obstacle_array.markers.append(lidar_obstacle)

            # S자 예외구간 작은 순두부 
            lidar_obstacle_threshold = Marker()
            lidar_obstacle_threshold.header.frame_id = "map"
            lidar_obstacle_threshold.id = i+100
            lidar_obstacle_threshold.type = lidar_obstacle_threshold.CYLINDER
            lidar_obstacle_threshold.action = lidar_obstacle.ADD
            lidar_obstacle_threshold.scale.x = self.distance_threshold_small * 2
            lidar_obstacle_threshold.scale.y = self.distance_threshold_small * 2 
            lidar_obstacle_threshold.scale.z = 0.2
            lidar_obstacle_threshold.pose.orientation.w = 1.0
            lidar_obstacle_threshold.color.r = 1.0
            lidar_obstacle_threshold.color.g = 0.0
            lidar_obstacle_threshold.color.b = 0.0
            lidar_obstacle_threshold.color.a = 0.2
            lidar_obstacle_threshold.pose.position.x = self.lidar_obstacle_info[i][0]
            lidar_obstacle_threshold.pose.position.y = self.lidar_obstacle_info[i][1]
            lidar_obstacle_threshold.pose.position.z = 0.0
            lidar_obstacle_threshold.lifetime = rospy.Duration(0.1)

            lidar_obstacle_big_threshold_array.markers.append(lidar_obstacle_threshold)

            # S자 예외구간 큰 순두부 
            lidar_obstacle_threshold = Marker()
            lidar_obstacle_threshold.header.frame_id = "map"
            lidar_obstacle_threshold.id = i+100
            lidar_obstacle_threshold.type = lidar_obstacle_threshold.CYLINDER
            lidar_obstacle_threshold.action = lidar_obstacle.ADD
            lidar_obstacle_threshold.scale.x = self.distance_threshold_big * 2
            lidar_obstacle_threshold.scale.y = self.distance_threshold_big * 2 
            lidar_obstacle_threshold.scale.z = 0.2
            lidar_obstacle_threshold.pose.orientation.w = 1.0
            lidar_obstacle_threshold.color.r = 0.0
            lidar_obstacle_threshold.color.g = 0.0
            lidar_obstacle_threshold.color.b = 1.0
            lidar_obstacle_threshold.color.a = 0.2
            lidar_obstacle_threshold.pose.position.x = self.lidar_obstacle_info[i][0]
            lidar_obstacle_threshold.pose.position.y = self.lidar_obstacle_info[i][1]
            lidar_obstacle_threshold.pose.position.z = 0.0
            lidar_obstacle_threshold.lifetime = rospy.Duration(0.1)

            lidar_obstacle_small_threshold_array.markers.append(lidar_obstacle_threshold)


        self.lidar_obstacle_pub.publish(lidar_obstacle_array)
        self.lidar_obstacle_big_threshold_pub.publish(lidar_obstacle_big_threshold_array)
        self.lidar_obstacle_small_threshold_pub.publish(lidar_obstacle_small_threshold_array)

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
        curvature_target_point.color.r = 1.0
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

    
if __name__ == '__main__':
    try:
        pure_pursuit_= PurePursuit()
    except rospy.ROSInterruptException:
        pass