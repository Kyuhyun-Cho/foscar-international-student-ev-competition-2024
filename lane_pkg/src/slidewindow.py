import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import *
from matplotlib.pyplot import *
import math
# float로 조향값 public

TOTAL_CNT = 50

class SlideWindow:


    
    def __init__(self):
        self.current_line = "DEFAULT"
        self.left_fit = None
        self.right_fit = None
        self.leftx = None
        self.rightx = None
        self.lhd = 240
        self.left_cnt = 25
        self.right_cnt = 25

        self.x_previous = 320



    def slidewindow(self, img):

        x_location = 320.0
        # init out_img, height, width        
        out_img = np.dstack((img, img, img)) * 255# deleted
        # out_img = img # added 
        height = img.shape[0]
        width = img.shape[1]

        # num of windows and init the height
        window_height = 15 # 7
        nwindows = 30 # 30
        
        # find nonzero location in img, nonzerox, nonzeroy is the array flatted one dimension by x,y 
        nonzero = img.nonzero()
        #print nonzero 
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        #print nonzerox
        # init data need to sliding windows
        margin = 40 
        minpix = 0 #10
        left_lane_inds = []
        right_lane_inds = []

        win_h1 = 300 
        win_h2 = 465
        win_l_w_l = 65 
        win_l_w_r = 210 # 205
        win_r_w_l = 430
        win_r_w_r = 575 # 550
        
        circle_height = 240

        # first location and segmenation location finder
        # draw line
        # 130 -> 150 -> 180
        pts_left = np.array([[win_l_w_l, win_h2], [win_l_w_l, win_h1], [win_l_w_r, win_h1], [win_l_w_r, win_h2]], np.int32)
        cv2.polylines(out_img, [pts_left], False, (0,255,0), 1)
        pts_right = np.array([[win_r_w_l, win_h2], [win_r_w_l, win_h1], [win_r_w_r, win_h1], [win_r_w_r, win_h2]], np.int32)
        cv2.polylines(out_img, [pts_right], False, (255,0,0), 1)

        pts_catch = np.array([[0, circle_height], [width, circle_height]], np.int32)
        cv2.polylines(out_img, [pts_catch], False, (0,120,120), 1)


        # indicies before start line(the region of pts_left)
        # 337 -> 310
        good_left_inds = ((nonzerox >= win_l_w_l) & (nonzeroy <= win_h2) & (nonzeroy > win_h1) & (nonzerox <= win_l_w_r)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_r_w_l) & (nonzeroy <= win_h2) & (nonzeroy > win_h1) & (nonzerox <= win_r_w_r)).nonzero()[0]

        # left line exist, lefty current init
        y_current = None
        x_current = None

        # check the minpix before left start line
        # if minpix is enough on left, draw left, then draw right depends on left
        # else draw right, then draw left depends on right

 
        if len(good_left_inds) > len(good_right_inds): 
       
            # self.current_line = "LEFT"
            line_flag = 1
            x_current = int(np.mean(nonzerox[good_left_inds]))
            y_current = int(np.mean(nonzeroy[good_left_inds]))
    
        elif len(good_left_inds) < len(good_right_inds):
 
            # self.current_line = "RIGHT"
            line_flag = 2
            x_current = int(np.mean(nonzerox[good_right_inds]))
            y_current = int(np.mean(nonzeroy[good_right_inds]))
            
        else:
            # print("Cant SEE!!!!")
            self.current_line = "MID"
            line_flag = 3   


        # it's just for visualization of the valid inds in the region: ind dot
        if line_flag == 1:
            for i in range(len(good_left_inds)):
                img = cv2.circle(out_img, (nonzerox[good_left_inds[i]], nonzeroy[good_left_inds[i]]), 1, (0,255,0), -1)
        elif line_flag == 2:
            for i in range(len(good_right_inds)):
                img = cv2.circle(out_img, (nonzerox[good_right_inds[i]], nonzeroy[good_right_inds[i]]), 1, (255,0,0), -1)
            

        


        # window sliding and draw
        for window in range(0, nwindows):
            if line_flag == 1: 
                # rectangle x,y range init
                win_y_low = y_current - (window + 1) * window_height
                win_y_high = y_current - (window) * window_height
                win_x_low = x_current - margin
                win_x_high = x_current + margin
                
                # draw rectangle
                # 0.33 is for width of the road
                cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 1)
                cv2.rectangle(out_img, (win_x_low + int(width * 0.55), win_y_low), (win_x_high + int(width * 0.55), win_y_high), (255, 0, 0), 1)
                
                # indicies of dots in nonzerox in one square
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
                
                # check num of indicies in square and put next location to current 
                if len(good_left_inds) > minpix:
                    x_current = int(np.mean(nonzerox[good_left_inds]))
                
                elif nonzeroy[left_lane_inds] != [] and nonzerox[left_lane_inds] != []:
                    p_left = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2) 
                    x_current = int(np.polyval(p_left, win_y_high))
                    
                # 338~344 is for recognize line which is yellow line in processed image(you can check in imshow)
                # print("win:", win_y_low)          
                          
                if circle_height - 10 <= win_y_low < circle_height + 10:
                    # 0.165 is the half of the road(0.33)
                    x_location = x_current + int(width * 0.275)
                    # self.x_previous = x_location
                    cv2.circle(out_img, (x_location, circle_height), 10, (0, 0, 255), 5)

            elif line_flag == 2: # change line from left to right above(if)
                
                win_y_low = y_current - (window + 1) * window_height
                win_y_high = y_current - (window) * window_height
                win_x_low = x_current - margin
                win_x_high = x_current + margin
                
                
                cv2.rectangle(out_img, (win_x_low - int(width * 0.55), win_y_low), (win_x_high - int(width * 0.55), win_y_high), (0, 255, 0), 1)
                cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (255, 0, 0), 1)
                
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

                if len(good_right_inds) > minpix:
                    x_current = int(np.mean(nonzerox[good_right_inds]))

                elif nonzeroy[right_lane_inds] != [] and nonzerox[right_lane_inds] != []:
                    p_right = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2) 
                    x_current = int(np.polyval(p_right, win_y_high))
                    
                if circle_height - 10 <= win_y_low < circle_height + 10:
                    # 0.165 is the half of the road(0.33)
                    x_location = x_current - int(width * 0.275) 
                    # self.x_previous = x_location
                    cv2.circle(out_img, (x_location, circle_height), 10, (0, 0, 255), 5)
            


            else : # can't see
                # print("Cant SEE!!!!")
                x_location = self.x_previous
                cv2.circle(out_img, (x_location, circle_height), 10, (0, 0, 255), 5)


            if x_location == 320:
                x_location = self.x_previous
                cv2.circle(out_img, (x_location, circle_height), 10, (0, 0, 255), 5)

            self.x_previous = x_location

        return out_img, x_location, self.current_line