# -*- coding: utf-8 -*-
import rospy
import rospkg
from nav_msgs.msg import Path,Odometry
from geometry_msgs.msg import PoseStamped,Point
from std_msgs.msg import Float64,Int16,Float32MultiArray
import numpy as np
from math import cos,sin,sqrt,pow,atan2,pi
import tf


class pathReader :  ## 텍스트 파일에서 경로를 출력 ##
    def __init__(self,pkg_name):
        rospack=rospkg.RosPack()
        self.file_path=rospack.get_path(pkg_name)



    def read_txt(self,file_name):
        full_file_name=self.file_path+"/path/wonju/"+file_name
        openFile = open(full_file_name, 'r')
        out_path=Path()

        target_velocity_array = []
        
        out_path.header.frame_id='map'
        line=openFile.readlines()
        for i in line :
            tmp=i.split()
            read_pose=PoseStamped()
            read_pose.pose.position.x=float(tmp[0])
            read_pose.pose.position.y=float(tmp[1])
            read_pose.pose.position.z=0
            read_pose.pose.orientation.x=0
            read_pose.pose.orientation.y=0
            read_pose.pose.orientation.z=0
            read_pose.pose.orientation.w=1
            out_path.poses.append(read_pose)

        openFile.close()
        return out_path, target_velocity_array ## 읽어온 경로를 global_path로 반환 ##
    

def findLocalPath(ref_path,status_msg): ## global_path와 차량의 status_msg를 이용해 현재waypoint와 local_path를 생성 ##
    out_path=Path()
    current_x=status_msg.position.x
    current_y=status_msg.position.y
    current_waypoint=0
    min_dis=float('inf')

    waypoint_counts = 50 # 기존 주행 코스 50 최적값.

    for i in range(len(ref_path.poses)) :
        dx=current_x - ref_path.poses[i].pose.position.x
        dy=current_y - ref_path.poses[i].pose.position.y
        dis=sqrt(dx*dx + dy*dy)
        if dis < min_dis :
            min_dis=dis
            current_waypoint=i


    if current_waypoint + waypoint_counts > len(ref_path.poses) :
        last_local_waypoint= len(ref_path.poses)
    else :
        last_local_waypoint=current_waypoint + waypoint_counts



    out_path.header.frame_id='map'
    for i in range(current_waypoint,last_local_waypoint) :
        tmp_pose=PoseStamped()
        tmp_pose.pose.position.x=ref_path.poses[i].pose.position.x
        tmp_pose.pose.position.y=ref_path.poses[i].pose.position.y
        tmp_pose.pose.position.z=ref_path.poses[i].pose.position.z
        tmp_pose.pose.orientation.x=0
        tmp_pose.pose.orientation.y=0
        tmp_pose.pose.orientation.z=0
        tmp_pose.pose.orientation.w=1
        out_path.poses.append(tmp_pose)

    return out_path,current_waypoint ## local_path와 waypoint를 반환 ##



class purePursuit: ## purePursuit 알고리즘 적용 ##
    def __init__(self):
        self.forward_point=Point()
        self.current_postion=Point()
        self.is_look_forward_point=False
        self.vehicle_length=3.0
        self.lfd=3
        self.min_lfd=1.3
        self.max_lfd=5.0
        self.steering=0
        
        self.is_obstacle_passed = False
        self.first_clock_wise = None

    def getPath(self,msg):
        self.path=msg  #nav_msgs/Path 
    
    
    def getEgoStatus(self, msg):

        self.current_vel=msg.velocity.x  #kph
        self.vehicle_yaw=(msg.heading)/180*pi   # rad
        self.current_postion.x=msg.position.x ## 차량의 현재x 좌표 ##
        self.current_postion.y=msg.position.y ## 차량의 현재y 좌표 ##
        self.current_postion.z=0.0 ## 차량의 현재z 좌표 ##



    def steering_angle(self, static_lfd=0): ## purePursuit 알고리즘을 이용한 Steering 계산 ## 
        vehicle_position=self.current_postion
        rotated_point=Point()
        self.is_look_forward_point= False

        
        for i in self.path.poses : # self.path == local_path 
            path_point=i.pose.position
            dx= path_point.x - vehicle_position.x
            dy= path_point.y - vehicle_position.y
            rotated_point.x=cos(self.vehicle_yaw)*dx + sin(self.vehicle_yaw)*dy
            rotated_point.y=sin(self.vehicle_yaw)*dx - cos(self.vehicle_yaw)*dy
           
 
            if rotated_point.x>0 :
                dis=sqrt(pow(rotated_point.x,2)+pow(rotated_point.y,2))
                
                if dis>= self.lfd :
                    self.lfd=self.current_vel * 0.12 # wonju
                    if self.lfd < self.min_lfd : 
                        self.lfd=self.min_lfd

                    elif self.lfd > self.max_lfd :
                        self.lfd=self.max_lfd

                    if static_lfd > 0:
                        self.lfd = static_lfd

                    # print("lfd : ", self.lfd)

                    self.forward_point=path_point
                    self.is_look_forward_point=True
                    
                    break
        
        theta=atan2(rotated_point.y,rotated_point.x)

        if self.is_look_forward_point :
            self.steering=atan2((2*self.vehicle_length*sin(theta)),self.lfd)*180/pi * -1 #deg
            return self.steering, self.forward_point.x, self.forward_point.y ## Steering 반환 ##
        else : 
            return 0, 0, 0
        

    def corner_estimation(self):
        vehicle_position=self.current_postion
        rotated_point=Point()
        
        last_path_point = self.path.poses[-1].pose.position
        dx = last_path_point.x - vehicle_position.x
        dy = last_path_point.y - vehicle_position.y

        rotated_point.x=cos(self.vehicle_yaw)*dx + sin(self.vehicle_yaw)*dy
        rotated_point.y=sin(self.vehicle_yaw)*dx - cos(self.vehicle_yaw)*dy
    
        self.far_foward_point = last_path_point

        corner_theta = abs(atan2(rotated_point.y,rotated_point.x))
        corner_theta_degree = corner_theta * 180 /pi

        return corner_theta_degree,self.far_foward_point.x,self.far_foward_point.y


    def getMinDistance(self, ref_path, obstacle_info, vehicle_status):
        
        min_distance = 99999
        min_path_coord= [0, 0]
        min_obstacle_coord = [0, 0]

        for obstacle in obstacle_info:

            for path_pos in ref_path.poses:

                distance_from_path= sqrt(pow(obstacle[0]-path_pos.pose.position.x,2)+pow(obstacle[1]-path_pos.pose.position.y,2))
                distance_from_vehicle = max(sqrt((obstacle[0]-vehicle_status.position.x)**2 + (obstacle[1]-vehicle_status.position.y)**2),0.1)
                if distance_from_path < min_distance:
                    min_distance = distance_from_path
                    min_path_coord = [path_pos.pose.position.x, path_pos.pose.position.y]
                    min_obstacle_coord = [obstacle[0], obstacle[1]]
                        

        return min_distance, min_path_coord, min_obstacle_coord
    

    def checkDynamicObstacle(self, clock_wise, min_distance):
        is_dynamic_obstacle = False    
        distance_threshold = 4.5 # 4.5
        
        if self.first_clock_wise != None:
            if self.is_obstacle_passed == False:
                if (self.first_clock_wise * clock_wise) < 0:
                    distance_threshold = 2.5
                    self.is_obstacle_passed = True
                else:
                    self.is_obstacle_passed = False
                
            elif self.is_obstacle_passed == True:
                distance_threshold = 2.5
        else:
            self.first_clock_wise = clock_wise

        if min_distance <= distance_threshold:
            is_dynamic_obstacle = True
        else:
            is_dynamic_obstacle = False

        return is_dynamic_obstacle, distance_threshold


def CCW(vehicle_coord, min_path_coord, min_obstacle_coord):
    cross_product = (min_path_coord[0] - vehicle_coord.position.x) * (min_obstacle_coord[1] - min_path_coord[1]) - (min_path_coord[1] - vehicle_coord.position.y) * (min_obstacle_coord[0] - min_path_coord[0])

    if cross_product > 0:
        return -1 # 시계 반대방향인 경우
    elif cross_product < 0:
        return 1  # 시계방향인 경우
    else:
        return 0
    

def rotateLiDAR2GPS(fusion_result, vehicle_status) :

    lidar_x_position = 3.45
    lidar_y_position = 0.0

    if len(fusion_result) == 0: 
        return []

    fusion_result_in_map = []

    theta = vehicle_status.heading / 180*pi # vehicle heading
    
    for bbox in fusion_result :
        x = bbox[0]
        y = bbox[1]

        new_x = (x+lidar_x_position)*cos(theta) - (y+lidar_y_position)*sin(theta) + vehicle_status.position.x
        new_y = (x+lidar_x_position)*sin(theta) + (y+lidar_y_position)*cos(theta) + vehicle_status.position.y

        fusion_result_in_map.append([new_x, new_y])

    return fusion_result_in_map
                    

                
            
