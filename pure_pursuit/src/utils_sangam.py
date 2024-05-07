# -*- coding: utf-8 -*-
import rospy
import rospkg
from nav_msgs.msg import Path,Odometry
from geometry_msgs.msg import PoseStamped,Point
from std_msgs.msg import Float64,Int16,Float32MultiArray
import numpy as np
from math import cos,sin,sqrt,pow,atan2,pi
import tf
import copy


class pathReader :  ## 텍스트 파일에서 경로를 출력 ##
    def __init__(self,pkg_name):
        rospack=rospkg.RosPack()
        self.file_path=rospack.get_path(pkg_name)



    def read_txt(self,file_name):
        full_file_name=self.file_path+"/path/sangam/"+file_name
        openFile = open(full_file_name, 'r')
        out_path_control=Path()

        out_path_control.header.frame_id='map'
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
            out_path_control.poses.append(read_pose)

        openFile.close()
        return out_path_control ## 읽어온 경로를 global_path로 반환 ##
    

def findLocalPath(ref_path,status_msg): ## global_path와 차량의 status_msg를 이용해 현재 waypoint와 local_path를 생성 ##
    out_path_control=Path()
    fifteen_past_path = Path()
    current_x=status_msg.position.x
    current_y=status_msg.position.y
    current_waypoint=0
    min_dis=float('inf')

    # waypoint_counts = 100 # 기존 주행 코스 50 최적값.
    waypoint_counts = 60

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


    
    out_path_control.header.frame_id='map'
    # fifteen_past_path.header.frame_id = 'map'
    for i in range(current_waypoint - 15, last_local_waypoint) :
        if i < 0:
            continue

        tmp_pose=PoseStamped()
        tmp_pose.pose.position.x=ref_path.poses[i].pose.position.x
        tmp_pose.pose.position.y=ref_path.poses[i].pose.position.y
        tmp_pose.pose.position.z=ref_path.poses[i].pose.position.z
        tmp_pose.pose.orientation.x=0
        tmp_pose.pose.orientation.y=0
        tmp_pose.pose.orientation.z=0
        tmp_pose.pose.orientation.w=1
        out_path_control.poses.append(tmp_pose)

    return out_path_control, current_waypoint, min_dis ## local_path와 waypoint를 반환 ##



class purePursuit : ## purePursuit 알고리즘 적용 ##
    def __init__(self):
        self.forward_point=Point()
        self.current_postion=Point()
        self.is_look_forward_point=False
        self.vehicle_length=4.635 # 4.6 # 3.0
        self.lfd=3
        self.min_lfd=5.0
        self.max_lfd=6.4
        self.steering=0
        
        self.is_obstacle_passed = False
        self.first_clock_wise = None

        self.previous_lattice_weights = [0, 0, 0]
        self.previous_selected_lane = 0

    def getPath(self,msg):
        self.path=msg  #nav_msgs/Path 
    
    
    def getEgoStatus(self, msg):

        self.current_vel=msg.velocity.x  #kph
        self.vehicle_yaw=(msg.heading)/180*pi   # rad
        self.current_postion.x=msg.position.x ## 차량의 현재x 좌표 ##
        self.current_postion.y=msg.position.y ## 차량의 현재y 좌표 ##
        self.current_postion.z=0.0 ## 차량의 현재z 좌표 ##



    def steeringAngle(self, static_lfd=0, current_waypoint=-1): ## purePursuit 알고리즘을 이용한 Steering 계산 ## 
        vehicle_position=self.current_postion
        rotated_point=Point()
        self.is_look_forward_point= False
        # print("self.lfd ",self.lfd)

        rotated_x_threshold = 0.0

        if (3662 <= current_waypoint <= 3693):
            rotated_x_threshold = 3.5
            static_lfd = 6
        
        
        for i in self.path.poses: # self.path == local_path 
            path_point=i.pose.position
            dx = path_point.x - vehicle_position.x
            dy = path_point.y - vehicle_position.y
            rotated_point.x=cos(self.vehicle_yaw)*dx + sin(self.vehicle_yaw)*dy
            rotated_point.y=sin(self.vehicle_yaw)*dx - cos(self.vehicle_yaw)*dy

            if rotated_point.x > rotated_x_threshold :
                dis=sqrt(pow(rotated_point.x,2)+pow(rotated_point.y,2))

                if static_lfd > 0:
                    self.lfd = static_lfd
                
                if dis >= self.lfd :
                    self.lfd=self.current_vel * 0.17 # sangam
                    if self.lfd < self.min_lfd : 
                        self.lfd=self.min_lfd 

                    elif self.lfd > self.max_lfd :
                        self.lfd=self.max_lfd
                    

                    self.forward_point=path_point
                    self.is_look_forward_point=True
                    
                    break
        
        theta=atan2(rotated_point.y,rotated_point.x)

        if self.is_look_forward_point :
            self.steering=atan2((2*self.vehicle_length*sin(theta)),self.lfd)*180/pi * -1 #deg
            return self.steering, self.forward_point.x, self.forward_point.y, self.lfd ## Steering 반환 ##
        else : 
            return 0, 0, 0, 0
        
    def blackoutSteeringAngle(self, black_out_local_path, motor_msg, static_lfd=7): ## purePursuit 알고리즘을 이용한 Steering 계산 ## 
        vehicle_position_x = -3.6  # 라이다 부착위치'
        vehicle_position_y = 0.0
        vehicle_yaw = 0.0
        lfd = static_lfd
        min_lfd = 5
        max_lfd = 10

        rotated_point=Point()
        self.is_look_forward_point= True
        # print("self.lfd ",self.lfd)

        rotated_point.x = black_out_local_path.poses[-1].pose.position.x
        rotated_point.y = black_out_local_path.poses[-1].pose.position.y

        lfd = sqrt(pow(rotated_point.x, 2) + pow(rotated_point.y, 2))
        # for i in black_out_local_path.poses : # self.path == local_path 
        #     path_point=i.pose.position
        #     dx= path_point.x - vehicle_position_x
        #     dy= path_point.y - vehicle_position_y
        #     rotated_point.x=cos(vehicle_yaw)*dx + sin(vehicle_yaw)*dy
        #     rotated_point.y=sin(vehicle_yaw)*dx - cos(vehicle_yaw)*dy
            
 
        #     if rotated_point.x>0 :
        #         dis=sqrt(pow(rotated_point.x,2)+pow(rotated_point.y,2))
                
        #         if dis>= lfd :
        #             lfd = motor_msg * 0.12 # sangam
        #             if lfd < min_lfd : 
        #                 lfd=min_lfd 

        #             elif lfd > max_lfd :
        #                 lfd=max_lfd

        #             if static_lfd > 0:
        #                 lfd = static_lfd

        #             print("lfd : ", lfd)

        #             self.forward_point=path_point
        #             self.is_look_forward_point=True
                    
        #             break
        
        # print("rotated_point.y: ", rotated_point.y)
        # print("rotated_point.x: ", rotated_point.x)
        theta=atan2(rotated_point.y,rotated_point.x)

        # print(f"theta: {theta}")

        if self.is_look_forward_point :
            self.steering=atan2((2*self.vehicle_length*sin(theta)),lfd)*180/pi#deg
            return self.steering, rotated_point.x, rotated_point.y ## Steering 반환 ##
        else : 
            return 0, 0, 0
        

    def estimateCurvature(self):
        vehicle_position = self.current_postion
        try:
            last_path_point = self.path.poses[-24].pose.position
        except:
            last_path_point = self.path.poses[-1].pose.position

        dx = last_path_point.x - vehicle_position.x
        dy = last_path_point.y - vehicle_position.y

        rotated_point=Point()
        rotated_point.x=cos(self.vehicle_yaw)*dx + sin(self.vehicle_yaw)*dy
        rotated_point.y=sin(self.vehicle_yaw)*dx - cos(self.vehicle_yaw)*dy
    
        self.far_foward_point = last_path_point

        corner_theta = abs(atan2(rotated_point.y,rotated_point.x))
        corner_theta_degree = corner_theta * 180 /pi

        return corner_theta_degree, self.far_foward_point.x, self.far_foward_point.y


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
    

    def checkDynamicObstacle(self, clock_wise, min_distance, current_waypoint):
        is_dynamic_obstacle = False    
        distance_threshold = 8.0 # 4.5

        # 오른쪽에서 오는애면 줄이기
        if self.first_clock_wise == 1:
            if 3380 <= current_waypoint <= 3700:
                distance_threshold = 3.5 # 3.0
            else:
                distance_threshold = 2.5

        
        if self.first_clock_wise != None:
            if self.is_obstacle_passed == False:
                if (self.first_clock_wise * clock_wise) < 0:
                    distance_threshold = 0.75 #
                    self.is_obstacle_passed = True
                else:
                    self.is_obstacle_passed = False
                
            elif self.is_obstacle_passed == True:
                distance_threshold = 0.75
        else:
            self.first_clock_wise = clock_wise

        if min_distance <= distance_threshold:
            is_dynamic_obstacle = True
        else:
            is_dynamic_obstacle = False

        return is_dynamic_obstacle, distance_threshold
    

    def isObstacleOnPath(self, ref_path, global_valid_obstacle, vehicle_status, current_waypoint):
        is_obstacle_on_path = False
        distance_object_to_car_list = []

        # 초반 ACC
        if current_waypoint <= 670:
            distance_threshold = 4.0
        elif 1617 <= current_waypoint <= 1741 or 1918 <= current_waypoint <= 1945:
            distance_threshold = 5.5

        # 패스 스위칭 
        elif 2580 <= current_waypoint <= 2982: 
            distance_threshold = 2.0
        elif 3436 <= current_waypoint <= 3664: 
            distance_threshold = 1.6
        elif 3690 <= current_waypoint: 
            distance_threshold = 1.2


        # 후반 ACC
        elif 2167 <= current_waypoint <= 2540:
            distance_threshold = 3.5
        else:
            distance_threshold = 2.7

        if len(global_valid_obstacle) > 0:

            for path_pos in ref_path.poses:
                for obstacle in global_valid_obstacle:

                    if 3690 <= current_waypoint and abs(obstacle[0]- vehicle_status.position.x) > 8:
                        
                        continue

                    dis= sqrt(pow(obstacle[0]-path_pos.pose.position.x,2)+pow(obstacle[1]-path_pos.pose.position.y,2))
                    distance_object_to_car = max(sqrt((obstacle[0]-vehicle_status.position.x)**2 + (obstacle[1]-vehicle_status.position.y)**2), 0.1)

                    distance_object_to_car_list.append(distance_object_to_car)

                    if dis <= distance_threshold :
                        is_obstacle_on_path = True
                    
                        
        
        return is_obstacle_on_path, distance_object_to_car_list, distance_threshold         

    ########################  lattice  ########################

    def latticePlanner(self, ref_path, global_vaild_object, vehicle_status, current_lane, lattice_area_num):
        if lattice_area_num == 'first':
            distance_threshold = 2.5  #1.5
        elif lattice_area_num == 'second':
            distance_threshold = 2.0
        elif lattice_area_num == 'third':
            distance_threshold = 2.0

        out_path_control=[]
        out_path_planning=[]

        selected_lane = -1
        lattice_current_lane = current_lane
        look_distance = int(vehicle_status.velocity.x * 0.43 + 8) #* 0.7)
        if lattice_area_num == 'third':
            look_distance -= 6
            if look_distance < 1:
                look_distance = 1

        # look_distance = 30
        
        if len(ref_path.poses)>look_distance :
            # control path
            end_of_local_path_idx = 15+look_distance
            if end_of_local_path_idx >= len(ref_path.poses):
                end_of_local_path_idx = -1

            global_ref_start_point=(ref_path.poses[15].pose.position.x,ref_path.poses[15].pose.position.y)
            global_ref_start_next_point=(ref_path.poses[16].pose.position.x,ref_path.poses[16].pose.position.y)
            global_ref_end_point=(ref_path.poses[end_of_local_path_idx].pose.position.x,ref_path.poses[end_of_local_path_idx].pose.position.y)
            
            theta=atan2(global_ref_start_next_point[1]-global_ref_start_point[1],global_ref_start_next_point[0]-global_ref_start_point[0])
            translation=[global_ref_start_point[0],global_ref_start_point[1]]

            t=np.array([[cos(theta), -sin(theta),translation[0]],[sin(theta),cos(theta),translation[1]],[0,0,1]])
            det_t=np.array([[t[0][0],t[1][0],-(t[0][0]*translation[0]+t[1][0]*translation[1])   ],[t[0][1],t[1][1],-(t[0][1]*translation[0]+t[1][1]*translation[1])   ],[0,0,1]])

            world_end_point=np.array([[global_ref_end_point[0]],[global_ref_end_point[1]],[1]])
            local_end_point=det_t.dot(world_end_point)

            world_start_point=np.array([[global_ref_start_point[0]],[global_ref_start_point[1]],[1]])
            # local_start_point=det_t.dot(world_start_point)

            # planning path
            global_ref_start_point_2=(ref_path.poses[0].pose.position.x,ref_path.poses[0].pose.position.y)
            global_ref_start_next_point_2=(ref_path.poses[1].pose.position.x,ref_path.poses[1].pose.position.y)
            global_ref_end_point_2=(ref_path.poses[0].pose.position.x,ref_path.poses[0].pose.position.y)
            
            theta_2=atan2(global_ref_start_next_point_2[1]-global_ref_start_point_2[1],global_ref_start_next_point_2[0]-global_ref_start_point_2[0])
            translation_2=[global_ref_start_point_2[0],global_ref_start_point_2[1]]

            t_2=np.array([[cos(theta_2), -sin(theta_2),translation_2[0]],[sin(theta_2),cos(theta_2),translation_2[1]],[0,0,1]])
            det_t_2=np.array([[t_2[0][0],t_2[1][0],-(t_2[0][0]*translation_2[0]+t_2[1][0]*translation_2[1])   ],[t_2[0][1],t_2[1][1],-(t_2[0][1]*translation_2[0]+t_2[1][1]*translation_2[1])   ],[0,0,1]])

            world_end_point_2=np.array([[global_ref_end_point_2[0]],[global_ref_end_point_2[1]],[1]])
            local_end_point_2=det_t_2.dot(world_end_point_2)

            world_start_point_2=np.array([[global_ref_start_point_2[0]],[global_ref_start_point_2[1]],[1]])
            # local_start_point_2=det_t_2.dot(world_start_point_2)

            # common
            world_ego_vehicle_position=np.array([[vehicle_status.position.x],[vehicle_status.position.y],[1]])
            local_ego_vehicle_position=det_t.dot(world_ego_vehicle_position)

            # lattice 간격

            if lattice_area_num == 'first':
                lane_off_set=[3.0, 0, -3.0]
            elif lattice_area_num == 'second':
                lane_off_set=[6.1, 3.2, 0]
            elif lattice_area_num == 'third':
                lane_off_set=[-3.1, -3.1, 0]
                
            local_lattice_points=[]
            local_lattice_points_2=[]

            ############################# lattice path 생성  #############################
            for i in range(len(lane_off_set)):
                # control path
                local_lattice_points.append([local_end_point[0][0], local_end_point[1][0]+lane_off_set[i], 1])

                # planning path
                local_lattice_points_2.append([local_end_point_2[0][0], local_end_point_2[1][0]+lane_off_set[i], 1])

            for end_point in local_lattice_points :
                lattice_path=Path()
                lattice_path.header.frame_id='map'
                x=[]
                y=[]
                x_interval=0.5  # 0.3
                xs=0
                xf=end_point[0]
                ps=local_ego_vehicle_position[1][0]
 
                pf=end_point[1]
                x_num=xf/x_interval

                for i in range(xs,int(x_num)) : 
                    x.append(i*x_interval)
                
                a=[0.0,0.0,0.0,0.0]
                a[0]=ps
                a[1]=0
                a[2]=3.0*(pf-ps)/(xf*xf)
                a[3]=-2.0*(pf-ps)/(xf*xf*xf)

                for i in x:
                    result=a[3]*i*i*i+a[2]*i*i+a[1]*i+a[0]
                    y.append(result)


                for i in range(0,len(y)) :
                    local_result=np.array([[x[i]],[y[i]],[1]])
                    global_result=t.dot(local_result)

                    read_pose=PoseStamped()
                    read_pose.pose.position.x=global_result[0][0]
                    read_pose.pose.position.y=global_result[1][0]
                    read_pose.pose.position.z=0
                    read_pose.pose.orientation.x=0
                    read_pose.pose.orientation.y=0
                    read_pose.pose.orientation.z=0
                    read_pose.pose.orientation.w=1
                    lattice_path.poses.append(read_pose)

                out_path_control.append(lattice_path)

            for end_point in local_lattice_points_2 :
                lattice_path=Path()
                lattice_path.header.frame_id='map'
                x=[]
                y=[]
                x_interval=0.5  # 0.3
                xs=0
                xf=end_point[0]
                ps=local_ego_vehicle_position[1][0]
                # ps=local_start_point[1][0]
 
                pf=end_point[1]
                x_num=xf/x_interval

                for i in range(xs,int(x_num)) : 
                    x.append(i*x_interval)
                
                a=[0.0,0.0,0.0,0.0]
                a[0]=ps
                a[1]=0
                a[2]=3.0*(pf-ps)/(xf*xf)
                a[3]=-2.0*(pf-ps)/(xf*xf*xf)

                for i in x:
                    result=a[3]*i*i*i+a[2]*i*i+a[1]*i+a[0]
                    y.append(result)


                for i in range(0,len(y)) :
                    local_result=np.array([[x[i]],[y[i]],[1]])
                    global_result=t.dot(local_result)

                    read_pose=PoseStamped()
                    read_pose.pose.position.x=global_result[0][0]
                    read_pose.pose.position.y=global_result[1][0]
                    read_pose.pose.position.z=0
                    read_pose.pose.orientation.x=0
                    read_pose.pose.orientation.y=0
                    read_pose.pose.orientation.z=0
                    read_pose.pose.orientation.w=1
                    lattice_path.poses.append(read_pose)

                out_path_planning.append(lattice_path)
            ############################# lattice path 생성  #############################    

            add_point_size = int(vehicle_status.velocity.x*2*3.6) + 15
            if add_point_size > len(ref_path.poses)-2:
                add_point_size = len(ref_path.poses)

            elif add_point_size < 10 :
                add_point_size = 10

            # print('add point',add_point_size)
            
            for i in range(14+look_distance,add_point_size):
                if i+1 < len(ref_path.poses):
                    tmp_theta=atan2(ref_path.poses[i+1].pose.position.y-ref_path.poses[i].pose.position.y,ref_path.poses[i+1].pose.position.x-ref_path.poses[i].pose.position.x)
                    
                    tmp_translation=[ref_path.poses[i].pose.position.x,ref_path.poses[i].pose.position.y]
                    tmp_t=np.array([[cos(tmp_theta), -sin(tmp_theta),tmp_translation[0]],[sin(tmp_theta),cos(tmp_theta),tmp_translation[1]],[0,0,1]])
                    tmp_det_t=np.array([[tmp_t[0][0],tmp_t[1][0],-(tmp_t[0][0]*tmp_translation[0]+tmp_t[1][0]*tmp_translation[1])   ],[tmp_t[0][1],tmp_t[1][1],-(tmp_t[0][1]*tmp_translation[0]+tmp_t[1][1]*tmp_translation[1])   ],[0,0,1]])

                    for lane_num in range(len(lane_off_set)) :
                        local_result=np.array([[0],[lane_off_set[lane_num]],[1]])
                        global_result=tmp_t.dot(local_result)

                        read_pose=PoseStamped()
                        read_pose.pose.position.x=global_result[0][0]
                        read_pose.pose.position.y=global_result[1][0]
                        read_pose.pose.position.z=0
                        read_pose.pose.orientation.x=0
                        read_pose.pose.orientation.y=0
                        read_pose.pose.orientation.z=0
                        read_pose.pose.orientation.w=1
                        out_path_control[lane_num].poses.append(read_pose)


            for i in range(0,add_point_size):
                if i+1 < len(ref_path.poses):
                    tmp_theta=atan2(ref_path.poses[i+1].pose.position.y-ref_path.poses[i].pose.position.y,ref_path.poses[i+1].pose.position.x-ref_path.poses[i].pose.position.x)
                    
                    tmp_translation=[ref_path.poses[i].pose.position.x,ref_path.poses[i].pose.position.y]
                    tmp_t=np.array([[cos(tmp_theta), -sin(tmp_theta),tmp_translation[0]],[sin(tmp_theta),cos(tmp_theta),tmp_translation[1]],[0,0,1]])
                    tmp_det_t=np.array([[tmp_t[0][0],tmp_t[1][0],-(tmp_t[0][0]*tmp_translation[0]+tmp_t[1][0]*tmp_translation[1])   ],[tmp_t[0][1],tmp_t[1][1],-(tmp_t[0][1]*tmp_translation[0]+tmp_t[1][1]*tmp_translation[1])   ],[0,0,1]])

                    for lane_num in range(len(lane_off_set)) :
                        local_result=np.array([[0],[lane_off_set[lane_num]],[1]])
                        global_result=tmp_t.dot(local_result)

                        read_pose=PoseStamped()
                        read_pose.pose.position.x=global_result[0][0]
                        read_pose.pose.position.y=global_result[1][0]
                        read_pose.pose.position.z=0
                        read_pose.pose.orientation.x=0
                        read_pose.pose.orientation.y=0
                        read_pose.pose.orientation.z=0
                        read_pose.pose.orientation.w=1
                        out_path_planning[lane_num].poses.append(read_pose)


            # lane_weight=[2, 1, 0] #reference path 
            # lane_weight = [w for w in range(len(lane_off_set), 0, -1)]
            if lattice_area_num == 'first':
                lane_weight = [2, 0, 1]  #1.5
            elif lattice_area_num == 'second':
                lane_weight = [2, 1, 0] 
            elif lattice_area_num == 'third':
                lane_weight = [1, 1, 0] 

            collision_bool=[False for _ in range(len(lane_off_set))]

            # lattice path 내 장애물 탐색하여 가중치 조절
            if len(global_vaild_object)>0:

                for obj in global_vaild_object :
                    
                    for path_num in range(len(out_path_planning)) :
                        
                        for path_pos in out_path_planning[path_num].poses : #path_pos = PoseStamped()
                            
                            dis= sqrt(pow(obj[0]-path_pos.pose.position.x,2)+pow(obj[1]-path_pos.pose.position.y,2))
                            dis_car_obj = max(sqrt((obj[0]-vehicle_status.position.x)**2 + (obj[1]-vehicle_status.position.y)**2),0.1)
                            # print(path_num, dis)

                            if dis <= distance_threshold :
                                collision_bool[path_num]=True
                                # lane_weight[path_num]=lane_weight[path_num]+100
                                lane_weight[path_num] += 2*dis**-1 * (1000/dis_car_obj)
                                # break
            
            selected_lane = lane_weight.index(min(lane_weight))
            
            all_lane_collision=True
        
        else :
            print("NO Reference Path")
            selected_lane=-1    

        return out_path_control, out_path_planning, selected_lane, distance_threshold


    ########################  lattice  ########################

# 1 3 5 7 6 4 2 0

def CCW(vehicle_coord, min_path_coord, min_obstacle_coord):
    cross_product = (min_path_coord[0] - vehicle_coord.position.x) * (min_obstacle_coord[1] - min_path_coord[1]) - (min_path_coord[1] - vehicle_coord.position.y) * (min_obstacle_coord[0] - min_path_coord[0])

    if cross_product > 0:
        return -1 # 시계 반대방향인 경우
    elif cross_product < 0:
        return 1  # 시계방향인 경우
    else:
        return 0
    

def rotateLiDAR2GPS(obstacle_array, vehicle_status, current_waypoint=-1, only_dynamic_obstacle=False) :

    lidar_x_position = 4.20 #1.44
    lidar_y_position = 0.0

    if len(obstacle_array) == 0: 
        return []

    fusion_result_in_map = []

    theta = vehicle_status.heading / 180*pi # vehicle heading

    for bbox in obstacle_array :
        x = bbox[0]
        y = bbox[1]
        id = bbox[3]

        # 동적 장애물 정보만 필요하면 동적 장애물 정보만 필터링해서 사용
        if only_dynamic_obstacle is True:
            if id != 2:  # id 1: PE_DRUM, id 2: Person
                continue

        # 특정 구간에서 우측 가로등 보고 차가 영원히 멈추는 문제 해결하기 위해서 차량 기준 오른쪽에 있는 장애물은 무시
        if (2005 <= current_waypoint < 2577): 
            if y <-1:
                continue

        new_x = (x+lidar_x_position)*cos(theta) - (y+lidar_y_position)*sin(theta) + vehicle_status.position.x
        new_y = (x+lidar_x_position)*sin(theta) + (y+lidar_y_position)*cos(theta) + vehicle_status.position.y

        fusion_result_in_map.append([new_x, new_y])

    return fusion_result_in_map
                    

                
