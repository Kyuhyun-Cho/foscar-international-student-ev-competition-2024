#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np
import math
from sensor_msgs.msg import PointCloud2, CompressedImage, Image
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from lidar_cam_fusion.msg import Float64Array, Float64Array2D
from lidar_object_detection.msg import PointInfo
from ultralytics_ros.msg import YoloResult


class CreateMatrix:
    def __init__(self, params_cam, params_lidar):
        global RT, proj_mtx
        
        self.params_cam = params_cam
        self.params_lidar = params_lidar
        self.width = params_cam["WIDTH"]
        self.height = params_cam["HEIGHT"]
        
        RT = self.transfromMTX_lidar2cam(self.params_lidar, self.params_cam)
        
        proj_mtx = self.project2img_mtx(self.params_cam)
    
    
    def translationMtx(self, x, y, z):
        
        M = np.array([[1,       0,      0,      x],
                      [0,       1,      0,      y],
                      [0,       0,      1,      z],
                      [0,       0,      0,      1]])
        
        return M


    def rotationMtx(self, yaw, pitch, roll):
        
        R_x = np.array([[1,     0,                  0,                  0],
                        [0,     math.cos(roll),     -math.sin(roll),    0],
                        [0,     math.sin(roll),     math.cos(roll),     0],
                        [0,     0,                  0,                  1]])
        
        R_y = np.array([[math.cos(pitch),       0,      math.sin(pitch),     0],
                        [0,                     1,      0,                   0],
                        [-math.sin((pitch)),    0,      math.cos(pitch),     0],
                        [0,                     0,      0,                   1]])
        
        R_z = np.array([[math.cos(yaw),      -math.sin(yaw),     0,     0],
                        [math.sin(yaw),      math.cos(yaw),      0,     0],
                        [0,                  0,                  1,     0],
                        [0,                  0,                  0,     1]])
        
        
        R = np.matmul(R_x, np.matmul(R_y, R_z)) # x, y, z 계산 순서가 중요함
        
        return R


    def transfromMTX_lidar2cam(self, params_lidar, params_cam):
        
        #Relative position of lidar w.r.t cam
        lidar_pos = [params_lidar.get(i) for i in (["X", "Y", "Z"])]
        cam_pos = [params_cam.get(i) for i in (["X", "Y", "Z"])]
        
        x_rel = cam_pos[0] - lidar_pos[0]
        y_rel = cam_pos[1] - lidar_pos[1]
        z_rel = cam_pos[2] - lidar_pos[2]
        
        R_T = np.matmul(self.translationMtx(x_rel, y_rel, z_rel), self.rotationMtx(np.deg2rad(-90.), 0., 0.))
        R_T = np.matmul(R_T, self.rotationMtx(0., 0., np.deg2rad(-90.)))
        
        #rotate and translate the coordinate of a lidar (역행렬)
        R_T = np.linalg.inv(R_T)
        
        return R_T


    def project2img_mtx(self, params_cam):
        
        #focal lengths
        fc_x = params_cam["WIDTH"]/(2*np.tan(np.deg2rad(params_cam["FOV"]/2)))
        fc_y = params_cam["WIDTH"]/(2*np.tan(np.deg2rad(params_cam["FOV"]/2)))

        # the center of image
        cx = params_cam["WIDTH"]/2
        cy = params_cam["HEIGHT"]/2
        # cy = 470/2
        
        # transformation matrix from 3D to 2D
        R_f = np.array([[fc_x,  0,      cx],
                        [0,     fc_y,   cy]])
        
        return R_f
    

class Pointcloud2ImageTransform:
    def __init__(self, params_cam, params_lidar):
        # global cap
        
        self.params_cam = params_cam
        self.params_lidar = params_lidar
        self.width = params_cam["WIDTH"]
        self.height = params_cam["HEIGHT"]
        
        ######################## USB Cam 사용시 ########################
        # cap = cv2.VideoCapture(0, cv2.CAP_V4L)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        ######################## USB Cam 사용시 ########################
        
        self.yolo_img = []
        # self.yolo_reulst =

        self.pointcloud = PointCloud2()

        self.bbox_cnts = 0
        self.x_mini = []
        self.y_mini = []
        self.z_mini = []
        self.x_maxi = []
        self.y_maxi = []
        self.z_maxi = []

        self.bbox_3d_position = []
        self.bbox_yolo_position = []

        self.bridge = CvBridge()
        
        
        # self.image_sub = rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.img_callback)
        self.yolo_image_sub = rospy.Subscriber("/yolo_image", Image, self.yolo_img_callback)
        self.yolo_result_sub = rospy.Subscriber("/yolo_result", YoloResult, self.yolo_result_callback)
        self.pointcloud_sub = rospy.Subscriber("/roi_raw", PointCloud2, self.pointcloud_callback)
        self.bbox_point_sub = rospy.Subscriber("/bbox_point_info", PointInfo, self.bbox_point_callback)
        
        self.fusion_image_pub = rospy.Publisher("/fusion_img", Image, queue_size=1)
        self.fusion_result_pub = rospy.Publisher("/fusion_result", Float64Array2D, queue_size=1)

        self.rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            
            try:
                if len(self.pointcloud.data) > 0 and len(self.yolo_img) > 0:
        
                    self.get_yolo_bbox_position()
                    self.get_bbox_3d_position()

                    # self.xyz_p = self.pointcloud_to_xyz(self.pointcloud)
                    
                    # # LiDAR 기준 점을 Camera 기준점으로 변환
                    # self.xyz_c = self.transform_lidar2cam(self.xyz_p)
                    
                    # # Camera 기준 점을 Camera 이미지 위의 점으로 투영
                    # self.xyi = self.project_pts2img(self.xyz_c)
                    
                    # self.publish_fusion_img(self.xyi, False)

                    fusion_img = self.yolo_img

                    if self.bbox_cnts > 0:
                        fusion_img = self.draw_bbox_to_img(fusion_img)
                    else:
                        fusion_result_array = Float64Array2D()
                        self.fusion_result_pub.publish(fusion_result_array)

                    fusion_img = self.bridge.cv2_to_imgmsg(fusion_img, encoding="bgr8")
        
                    self.fusion_image_pub.publish(fusion_img) 


                # Pointclouds 아무 것도 없을 때는 그냥 카메라 이미지만 Publish
                elif len(self.pointcloud.data) == 0 and len(self.yolo_img) > 0:
                    imgmsg_without_pc = self.bridge.cv2_to_imgmsg(self.yolo_img,encoding="bgr8")
            
                    self.fusion_image_pub.publish(imgmsg_without_pc) 
                    print("Waiting for pointcloud")

                elif len(self.pointcloud.data) > 0 and len(self.yolo_img) == 0:
                    print("Waiting for yolo img")

                else:
                    print("Waiting for pointcloud & yolo img")

                self.rate.sleep()

            except:
                print("Something wrong, but I don't know")
                self.rate.sleep()
                pass


    def yolo_img_callback(self, msg):
        self.yolo_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        

    def yolo_result_callback(self, msg):
        bbox_list = [detection.bbox for detection in msg.detections.detections]
        result_list = [detection.results for detection in msg.detections.detections]

        self.center_x_list = [bbox.center.x for bbox in bbox_list]
        self.center_y_list = [bbox.center.y for bbox in bbox_list]

        self.size_x_list = [bbox.size_x for bbox in bbox_list]
        self.size_y_list = [bbox.size_y for bbox in bbox_list]

        self.id_list = [result[0].id for result in result_list]


    def pointcloud_callback(self, msg):
        self.pointcloud = msg
    
        
    def bbox_point_callback(self, msg):
        self.bbox_cnts = msg.bboxCounts
        
        self.x_mini = list(msg.xMini[:self.bbox_cnts])
        self.y_mini = list(msg.yMini[:self.bbox_cnts])
        self.z_mini = list(msg.zMini[:self.bbox_cnts])
        
        self.x_maxi = list(msg.xMaxi[:self.bbox_cnts])
        self.y_maxi = list(msg.yMaxi[:self.bbox_cnts])
        self.z_maxi = list(msg.zMaxi[:self.bbox_cnts])


    def get_yolo_bbox_position(self):
        self.bbox_yolo_position = []

        for i in range(len(self.id_list)):
            x1 = self.center_x_list[i] - self.size_x_list[i]/2
            y1 = self.center_y_list[i] - self.size_y_list[i]/2
            x2 = self.center_x_list[i] + self.size_x_list[i]/2
            y2 = self.center_y_list[i] + self.size_y_list[i]/2
            id = self.id_list[i]

            position = [x1, y1, x2, y2, id]
            self.bbox_yolo_position.append(position)


    def get_bbox_3d_position(self):

        self.bbox_3d_position = []

        for i in range(self.bbox_cnts):
            x = (self.x_mini[i] + self.x_maxi[i]) / 2
            y = (self.y_mini[i] + self.y_maxi[i]) / 2
            z = (self.z_mini[i] + self.z_maxi[i]) / 2
            position = [x, y, z]
            self.bbox_3d_position.append(position)


    def pointcloud_to_xyz(self, pointclouds):
        
        xyz_p = np.array(list(pc2.read_points(pointclouds, skip_nans=True)))
  
        xyz_p = xyz_p[:, :3]

        return xyz_p
        

        
    def transform_lidar2cam(self, xyz_p):
        
        xyz_c = np.matmul(np.concatenate([xyz_p, np.ones((xyz_p.shape[0], 1))], axis=1), RT.T)
        
        return xyz_c
    

    def project_pts2img(self, xyz_c, crop=True):
        
        xyz_c = xyz_c.T
        
        xc, yc, zc = xyz_c[0,:].reshape([1,-1]), xyz_c[1,:].reshape([1,-1]), xyz_c[2,:].reshape([1,-1])
        
        xn, yn = xc/(zc+0.0001), yc/(zc+0.0001)
        
        xyi = np.matmul(proj_mtx, np.concatenate([xn, yn, np.ones_like(xn)], axis=0))
        
        xyi = xyi[0:2,:].T
        
        if crop:
            xyi = self.crop_pts(xyi)
        else:
            pass
            
        return xyi


    def crop_pts(self, xyi):
        
        xyi = xyi[np.logical_and(xyi[:, 0]>=0, xyi[:, 0]<self.width), :]
        xyi = xyi[np.logical_and(xyi[:, 1]>=0, xyi[:, 1]<self.height), :]
        
        return xyi
    
    
    def crop_bbox_pts(self, bbox_xyi):
        bbox_xyi[:, 0] = np.where(bbox_xyi[:, 0] > 1279, 1279, np.where(bbox_xyi[:, 0] < 0, 0, bbox_xyi[:, 0]))
        bbox_xyi[:, 1] = np.where(bbox_xyi[:, 1] > 719, 719, np.where(bbox_xyi[:, 1] < 0, 0, bbox_xyi[:, 1]))

        return bbox_xyi


    def publish_fusion_img(self, xyi, with_pointcloud=False):
        # pointcloud 없이 boundingbox만 시각화

        fusion_img = self.yolo_img

        if with_pointcloud == False:
            if self.bbox_cnts > 0:
                fusion_img = self.draw_bbox_to_img(fusion_img)
            else:
                fusion_result_array = Float64Array2D()
                self.fusion_result_pub.publish(fusion_result_array)
    
                
        # pointcloud와 boundingbox 둘 다 시각화
        else:
            xi, yi = xyi[:, 0].reshape([1,-1]), xyi[:,1].reshape([1,-1])
            xi = xi.flatten()
            yi = yi.flatten()

            fusion_img = self.draw_pointcloud_to_img(fusion_img, xi, yi)

            if self.bbox_cnts > 0:
                fusion_img = self.draw_bbox_to_img(fusion_img)
        
        fusion_img = self.bridge.cv2_to_imgmsg(fusion_img, encoding="bgr8")
    
        self.fusion_image_pub.publish(fusion_img) 
        
        ######################## USB Cam 사용시 ########################
        # ret, cv_image = self.cap.read()
        
        # if ret:
        #     result = self.draw_pointcloud_to_img(cv_image, xi, yi)
        #     bridge = cv2.CvBrdige()
        #     self.fusion_image_pub.publish(bridge.cv2_to_imgmsg(result, "bgr8"))
        # else:
        #     print("There is no data")
        ######################## USB Cam 사용시 ########################

        
    def draw_pointcloud_to_img(self, img, xi, yi):
    
        point_np = img

        for ctr in zip(xi, yi):
            center = (int(ctr[0]), int(ctr[1]))
            point_np = cv2.circle(point_np, center, 1, (0,255, 0), -1)
            
        return point_np


    def draw_bbox_to_img(self, img):
        
        self.bbox_xyz_p = self.merge_bbox_points(self.x_mini, self.y_mini, self.z_mini, self.x_maxi, self.y_maxi, self.z_maxi)

        self.bbox_xyz_c = self.transform_bbox2cam(self.bbox_xyz_p)

        self.bbox_2d_xyi = self.project_bbox2img(self.bbox_xyz_c)
        
        self.bbox_2d_position = self.bbox_2d_xyi.reshape(-1, 4)

        bbox_np = img
        
        self.fusion_bbox_position = self.calculate_iou(self.bbox_yolo_position, self.bbox_2d_position)

        fusion_result_array = Float64Array2D()

        for fusion_result in self.fusion_bbox_position:
            fusion_result_array.bboxes.append(Float64Array(bbox=fusion_result))
        
        self.fusion_result_pub.publish(fusion_result_array)

        for i in range(0, self.bbox_cnts*2, 2): # 라이다 bbox 이미지에 투영
            
            bbox_np = cv2.rectangle(bbox_np, (int(self.bbox_2d_xyi[i][0]), 
                                              int(self.bbox_2d_xyi[i][1])), 
                                              (int(self.bbox_2d_xyi[i+1][0]), 
                                               int(self.bbox_2d_xyi[i+1][1])), 
                                               (0, 0, 255), 
                                               3)

        for i in range(len(self.fusion_bbox_position)): # 센서퓨전된 bbox 이미지에 투영
            
            bbox_np = cv2.rectangle(bbox_np, (self.fusion_bbox_position[i][0].astype(int), 
                                              self.fusion_bbox_position[i][1].astype(int)), 
                                              (self.fusion_bbox_position[i][2].astype(int), 
                                               self.fusion_bbox_position[i][3].astype(int)), 
                                               (0, 255, 0), 
                                               -1)

        return bbox_np
 
    

    def merge_bbox_points(self, x_mini, y_mini, z_mini, x_maxi, y_maxi, z_maxi):
        
        bbox_xyz_p = []
 
        for i in range(self.bbox_cnts):
            bbox_xyz_p.append([x_mini[i], y_mini[i], z_mini[i]])
            bbox_xyz_p.append([x_maxi[i], y_maxi[i], z_maxi[i]])
        
        bbox_xyz_p = np.array(bbox_xyz_p)
        
        return bbox_xyz_p  
    
        
    def transform_bbox2cam(self, bbox_xyz_p):

        bbox_xyz_c = np.matmul(np.concatenate([bbox_xyz_p, np.ones((bbox_xyz_p.shape[0], 1))], axis=1), RT.T)

        return bbox_xyz_c
    

    def project_bbox2img(self, bbox_xyz_c, crop=True):
        
        bbox_xyz_c = bbox_xyz_c.T
        
        bbox_xc, bbox_yc, bbox_zc = bbox_xyz_c[0,:].reshape([1,-1]), bbox_xyz_c[1,:].reshape([1,-1]), bbox_xyz_c[2,:].reshape([1,-1])
        
        bbox_xn, bbox_yn = bbox_xc/(bbox_zc+0.0001), bbox_yc/(bbox_zc+0.0001)
        
        bbox_xyi = np.matmul(proj_mtx, np.concatenate([bbox_xn, bbox_yn, np.ones_like(bbox_xn)], axis=0))
        
        bbox_xyi = bbox_xyi[0:2,:].T

        if crop:
            bbox_xyi = self.crop_bbox_pts(bbox_xyi)
        else:
            pass
        
        return bbox_xyi
    

    def calculate_iou(self, camera_bboxes, lidar_2d_bboxes):
        final_bboxes = np.empty((0,8)) # 2D_x1, 2D_y1, 2D_x2, 2D_y2, 3D_x, 3D_y, 3D_z, id

        # for camera_bbox in camera_bboxes:
        for i in range(len(camera_bboxes)):
            c_x1, c_y1, c_x2, c_y2, id = camera_bboxes[i]

            # for lidar_2d_bbox in lidar_2d_bboxes:
            for j in range(len(lidar_2d_bboxes)):
                l_x2, l_y2, l_x1, l_y1 = lidar_2d_bboxes[j]

                # Calculate intersection area
                x_left   = max(c_x1, l_x1)
                y_top    = max(c_y1, l_y1)
                x_right  = min(c_x2, l_x2)
                y_bottom = min(c_y2, l_y2)
                
                if x_right < x_left or y_bottom < y_top:
                    # No intersection
                    continue
                else:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)

                    # Calculate union area
                    camera_area_box = (c_x2 - c_x1) * (c_y2 - c_y1)
                    lidar_area_box = (l_x2 - l_x1) * (l_y2 - l_y1)
                    union_area = camera_area_box + lidar_area_box - intersection_area
                    
                    # Calculate IoU
                    # iou = intersection_area / union_area 
                    iou = intersection_area / lidar_area_box

                    print ("IoU:", iou)
                    if (iou >= 0.1):
                        intersection_info = np.array([[x_left, y_top, x_right, y_bottom, self.bbox_3d_position[j][0], self.bbox_3d_position[j][1], self.bbox_3d_position[j][2], id]])
                        final_bboxes = np.concatenate((final_bboxes, intersection_info), axis=0)
                        break

        return final_bboxes

        
if __name__ == '__main__':
    
    rospy.init_node('lidar_cam_calibration', anonymous=True)
    
    ###################################### 센서 파라미터 세팅 ######################################
    params_cam = {"WIDTH": 1280, "HEIGHT": 720, "FOV": 90, "X": 3.45, "Y": 0.00, "Z": 0.69}
    params_lidar = {"X": 3.45, "Y": 0.00, "Z": 0.60}
    ###################################### 센서 파라미터 세팅 ######################################
    
    prepare_matrix = CreateMatrix(params_cam, params_lidar)
    
    try:
        lidar2cam_transformer = Pointcloud2ImageTransform(params_cam, params_lidar)
    except rospy.ROSInterruptException:
        pass