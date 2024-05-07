import cv2
import numpy as np
import math
from scipy.interpolate import *
from matplotlib.pyplot import *


class SlideWindow:

    def __init__(self):

        self.current_line = "DEFAULT"
        self.left_fit = None
        self.right_fit = None
        self.leftx = None
        self.rightx = None
        self.left_cnt = 25
        self.right_cnt = 25

        self.x_previous = 320



    def slidewindow(self, img, is_detected):

        x_location = None
        out_img = np.dstack((img, img, img)) * 255 
        width = img.shape[1]
        # yaw = yaw
        window_height = 15
        nwindows = 10
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 20 
        minpix = 0 
        left_lane_inds = []
        right_lane_inds = []

        if is_detected == True :
            win_h1 = 325
            win_h2 = 480
            win_l_w_l = 220
            win_l_w_r = 260
            win_r_w_l = 380
            win_r_w_r = 420 

        else :
            win_h1 = 325
            win_h2 = 480
            win_l_w_l = 190
            win_l_w_r = 290
            win_r_w_l = 350
            win_r_w_r = 450

        between_distance = 0.25
        half_between_distance = 0.5 * between_distance

        decision_distance = 480
        pts_left = np.array([[win_l_w_l, win_h2], [win_l_w_l, win_h1], [win_l_w_r, win_h1], [win_l_w_r, win_h2]], np.int32)
        cv2.polylines(out_img, [pts_left], False, (0,255,0), 1)
        pts_right = np.array([[win_r_w_l, win_h2], [win_r_w_l, win_h1], [win_r_w_r, win_h1], [win_r_w_r, win_h2]], np.int32)
        cv2.polylines(out_img, [pts_right], False, (255,0,0), 1)

        pts_catch = np.array([[0, decision_distance], [width, decision_distance]], np.int32)
        cv2.polylines(out_img, [pts_catch], False, (0,120,120), 1)
        good_left_inds = ((nonzerox >= win_l_w_l) & (nonzeroy <= win_h2) & (nonzeroy > win_h1) & (nonzerox <= win_l_w_r)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_r_w_l) & (nonzeroy <= win_h2) & (nonzeroy > win_h1) & (nonzerox <= win_r_w_r)).nonzero()[0]
        y_current = None
        x_current = None

        # if yaw <- 40 and len(good_right_inds) >= 70 :
        #     self.current_line = "RIGHT"
        #     line_flag = 2
        #     x_current = int(np.mean(nonzerox[good_right_inds]))
        #     y_current = int(np.max(nonzeroy[good_right_inds]))
        if len(good_left_inds) > len(good_right_inds):      
            self.current_line = "LEFT"
            line_flag = 1
            x_current = int(np.mean(nonzerox[good_left_inds]))
            y_current = int(np.mean(nonzeroy[good_left_inds]))
            max_y = y_current
        elif len(good_right_inds) > len(good_left_inds):
        
            self.current_line = "RIGHT"
            line_flag = 2
            x_current = int(np.mean(nonzerox[good_right_inds]))
            y_current = int(np.max(nonzeroy[good_right_inds]))
        else:
            self.current_line = "MID"
            line_flag = 3   

        if line_flag == 1:
            for i in range(len(good_left_inds)):
                    img = cv2.circle(out_img, (nonzerox[good_left_inds[i]], nonzeroy[good_left_inds[i]]), 1, (0,255,0), -1)
        elif line_flag == 2:
            for i in range(len(good_right_inds)):
                    img = cv2.circle(out_img, (nonzerox[good_right_inds[i]], nonzeroy[good_right_inds[i]]), 1, (255,0,0), -1)
            
        for window in range(0, nwindows):
            if line_flag == 1: 
                win_y_low = y_current - (window + 1) * window_height
                win_y_high = y_current - (window) * window_height
                win_x_low = x_current - margin
                win_x_high = x_current + margin
                cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 1)
                cv2.rectangle(out_img, (win_x_low + int(width * between_distance), win_y_low), (win_x_high + int(width * between_distance), win_y_high), (255, 0, 0), 1)
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
                if len(good_left_inds) > minpix:
                    x_current = int(np.mean(nonzerox[good_left_inds]))
                
                elif nonzeroy[left_lane_inds] != [] and nonzerox[left_lane_inds] != []:
                    p_left = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2) 
                    x_current = int(np.polyval(p_left, win_y_high))                    
                if win_y_low >= 338 and win_y_low < 348:
                    x_location = x_current + int(width * half_between_distance) + 7
                    cv2.circle(out_img, (x_location, decision_distance), 10, (0, 0, 255), 5)

            elif line_flag == 2:
                win_y_low = y_current - (window + 1) * window_height
                win_y_high = y_current - (window) * window_height
                win_x_low = x_current - margin
                win_x_high = x_current + margin
                cv2.rectangle(out_img, (win_x_low - int(width * between_distance), win_y_low), (win_x_high - int(width * between_distance), win_y_high), (0, 255, 0), 1)
                cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (255, 0, 0), 1)
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

                if len(good_right_inds) > minpix:
                    x_current = int(np.mean(nonzerox[good_right_inds]))

                elif nonzeroy[right_lane_inds] != [] and nonzerox[right_lane_inds] != []:
                    p_right = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2) 
                    x_current = int(np.polyval(p_right, win_y_high))
                if win_y_low >= 338 and win_y_low < 348:
                    x_location = x_current - int(width * half_between_distance) 
                    cv2.circle(out_img, (x_location, decision_distance), 10, (0, 0, 255), 5)
            


            # else :
            #     x_location = self.x_previous
            #     cv2.circle(out_img, (x_location, decision_distance), 10, (0, 0, 255), 5)


            # if x_location == 320:
            #     x_location = self.x_previous
            #     cv2.circle(out_img, (x_location, decision_distance), 10, (0, 0, 255), 5)
            # self.x_previous = x_location


        return out_img, x_location, self.current_line