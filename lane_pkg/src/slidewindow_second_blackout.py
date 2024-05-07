import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import *
from matplotlib.pyplot import *
import math
# float로 조향값 public

TOTAL_CNT = 50

class SlideWindow2:
    
    def __init__(self):
        self.current_line = "DEFAULT"
        self.left_fit = None
        self.right_fit = None
        self.leftx = None
        self.rightx = None
        self.lhd = 360
        self.left_cnt = 25
        self.right_cnt = 25

        self.x_previous = 640


        self.x_min = 0 # 330
        self.x_max = 1280 # 930

    def protected_point(self, x_location):
        if x_location <= self.x_min:
            print("Detect left outlier point", x_location)
            x_location = self.x_min
        elif x_location >= self.x_max:
            print("Detect right outlier point", x_location)
            x_location = self.x_max

        return x_location
        


    def slidewindow(self, img):

        x_location = 640
        # init out_img, height, width        
        out_img = np.dstack((img, img, img)) * 255# deleted
        # out_img = img # added 
        height = img.shape[0]
        width = img.shape[1]

        # num of windows and init the height
        window_height = 30 # 7
        nwindows = 20 # 30
        
        # find nonzero location in img, nonzerox, nonzeroy is the array flatted one dimension by x,y 
        nonzero = img.nonzero()
        #print nonzero 
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        #print nonzerox
        # init data need to sliding windows
        margin = 50
        minpix = 100 #10
        left_lane_inds = []
        right_lane_inds = []

        win_h1 = 200#90 #300 
        win_h2 = 720
        win_l_w_l = 200 #225 #130 
        win_l_w_r = 470 + 10 #470
        win_r_w_l = 1280-420-60#1000 # 1280 - 340
        win_r_w_r = 1280-150#1210 # 1280 - 130 # 550
        
        circle_height = 360

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

        x_current_left = None
        y_current_left = None

        x_current_right = None
        y_current_right = None

        # check the minpix before left start line
        # if minpix is enough on left, draw left, then draw right depends on left
        # else draw right, then draw left depends on right
        
        road_width = 0.5
        min_pixel_cnt = 300

        if len(good_left_inds) > min_pixel_cnt and len(good_right_inds) > min_pixel_cnt:
            x_current_left = int(np.mean(nonzerox[good_left_inds]))
            y_current_left = int(np.max(nonzeroy[good_left_inds]))
    
            x_current_right = int(np.mean(nonzerox[good_right_inds]))
            y_current_right = int(np.max(nonzeroy[good_right_inds]))

    
            win_l_w_l = x_current_left - 30 #225 #130 
            win_l_w_r = x_current_left + 30
            win_r_w_l = x_current_right - 30
            win_r_w_r = x_current_right + 30
            pts_left = np.array([[win_l_w_l, win_h2], [win_l_w_l, win_h1], [win_l_w_r, win_h1], [win_l_w_r, win_h2]], np.int32)
            cv2.polylines(out_img, [pts_left], False, (0,255,0), 1)
            pts_right = np.array([[win_r_w_l, win_h2], [win_r_w_l, win_h1], [win_r_w_r, win_h1], [win_r_w_r, win_h2]], np.int32)
            cv2.polylines(out_img, [pts_right], False, (255,0,0), 1)

            pts_catch = np.array([[0, circle_height], [width, circle_height]], np.int32)
            cv2.polylines(out_img, [pts_catch], False, (0,120,120), 1)



            # 차선 색칠하기
            for i in range(len(good_left_inds)):
                img = cv2.circle(out_img, (nonzerox[good_left_inds[i]], nonzeroy[good_left_inds[i]]), 1, (0,255,0), -1)

            for i in range(len(good_right_inds)):
                img = cv2.circle(out_img, (nonzerox[good_right_inds[i]], nonzeroy[good_right_inds[i]]), 1, (255,0,0), -1)

            for window in range(0, nwindows):
                # 왼쪽 차선
                win_x_low_left = x_current_left - margin
                win_y_low_left = y_current_left - (window + 1) * window_height
                
                win_x_high_left = x_current_left + margin
                win_y_high_left = y_current_left - (window) * window_height

                # 오른쪽 차선
                win_x_low_right = x_current_right - margin
                win_y_low_right = y_current_right - (window + 1) * window_height

                win_x_high_right = x_current_right + margin
                win_y_high_right = y_current_right - (window) * window_height

                cv2.rectangle(out_img, (win_x_low_left, win_y_low_left), (win_x_high_left, win_y_high_left), (0, 255, 0), 1)
                cv2.rectangle(out_img, (win_x_low_right, win_y_low_right), (win_x_high_right, win_y_high_right), (255, 0, 0), 1)


                good_left_inds = ((nonzeroy >= win_y_low_left) & (nonzeroy < win_y_high_left) & (nonzerox >= win_x_low_left) & (nonzerox < win_x_high_left)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low_right) & (nonzeroy < win_y_high_right) & (nonzerox >= win_x_low_right) & (nonzerox < win_x_high_right)).nonzero()[0]

                # 각 슬라이딩 윈도우 내 픽셀 개수 계산해서 임계값 이상이면 다음으로 넘어가고, 임계값 이하이면 보간법으로 예측해서 도로 곡률에 맞게 예상 윈도우 그리기
                
                
                if len(good_left_inds) > minpix:
                    x_current_left = int(np.mean(nonzerox[good_left_inds]))
                    
                elif len(nonzeroy[left_lane_inds]) > 2 and len(nonzerox[left_lane_inds]) > 2:
                    p_left = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2) 
                    x_current_left = int(np.polyval(p_left, win_y_high_left))

                else:
                    pass
                    
                
                if len(good_right_inds) > minpix:
                    x_current_right = int(np.mean(nonzerox[good_right_inds]))
                
                elif len(nonzeroy[right_lane_inds]) > 2 and len(nonzerox[right_lane_inds]) > 2:
                    p_right = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2) 
                    x_current_right = int(np.polyval(p_right, win_y_high_right))

                else:
                    pass
            

                # 왼쪽 오른쪽 차선 기반으로 추종점 계산하기
                if (circle_height-10 <= win_y_low_left < circle_height+10):
                    
                    x_location = (x_current_left + x_current_right) // 2

                    x_location = self.protected_point(x_location)

                    cv2.circle(out_img, (x_location, circle_height), 10, (0, 0, 255), -1)

                elif (circle_height-10 <= win_y_low_right < circle_height+10):
                    
                    x_location = (x_current_left + x_current_right) // 2
                    x_location = self.protected_point(x_location)


                    cv2.circle(out_img, (x_location, circle_height), 10, (0, 0, 255), -1)

                # left_lane_inds.extend(good_left_inds)
                # right_lane_inds.extend(good_right_inds)

            

        elif len(good_left_inds) > min_pixel_cnt and len(good_right_inds) < min_pixel_cnt:
            x_current_left = int(np.mean(nonzerox[good_left_inds]))
            y_current_left = int(np.max(nonzeroy[good_left_inds]))

            win_l_w_l = x_current_left - 30 #225 #130 
            win_l_w_r = x_current_left + 30
            win_r_w_l = 1280-420-60#1000 # 1280 - 340
            win_r_w_r = 1280-150#1210 # 1280 - 130 # 550
            pts_left = np.array([[win_l_w_l, win_h2], [win_l_w_l, win_h1], [win_l_w_r, win_h1], [win_l_w_r, win_h2]], np.int32)
            cv2.polylines(out_img, [pts_left], False, (0,255,0), 1)
            pts_right = np.array([[win_r_w_l, win_h2], [win_r_w_l, win_h1], [win_r_w_r, win_h1], [win_r_w_r, win_h2]], np.int32)
            cv2.polylines(out_img, [pts_right], False, (255,0,0), 1)

            pts_catch = np.array([[0, circle_height], [width, circle_height]], np.int32)
            cv2.polylines(out_img, [pts_catch], False, (0,120,120), 1)
        
            for i in range(len(good_left_inds)):
                img = cv2.circle(out_img, (nonzerox[good_left_inds[i]], nonzeroy[good_left_inds[i]]), 1, (0,255,0), -1)

            for window in range(0, nwindows):
                # 왼쪽 차선
                win_y_low = y_current_left - (window + 1) * window_height
                win_y_high = y_current_left - (window) * window_height
                win_x_low = x_current_left - margin
                win_x_high = x_current_left + margin
                
                # draw rectangle
                # 0.33 is for width of the road
                cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 1)
                cv2.rectangle(out_img, (win_x_low + int(width * road_width), win_y_low), (win_x_high + int(width * road_width), win_y_high), (255, 0, 0), 1)
                
                # indicies of dots in nonzerox in one square
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

                # check num of indicies in square and put next location to current 
                
          
                if len(good_left_inds) > minpix:
                    x_current_left = int(np.mean(nonzerox[good_left_inds]))
                
                elif len(nonzeroy[left_lane_inds]) > 2 and len(nonzerox[left_lane_inds]) > 2:
                    p_left = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2) 
                    x_current_left = int(np.polyval(p_left, win_y_high))

                else:
                    pass
    
                          
                if circle_height - 10 <= win_y_low < circle_height + 10:

                
                    x_location = x_current_left + int(width * road_width/2)
                    x_location = self.protected_point(x_location)
                    
                    cv2.circle(out_img, (x_location, circle_height), 10, (0, 255, 0), -1)

                left_lane_inds.extend(good_left_inds)

            

        elif len(good_left_inds) < min_pixel_cnt and len(good_right_inds) > min_pixel_cnt:
            x_current_right = int(np.mean(nonzerox[good_right_inds]))
            y_current_right = int(np.max(nonzeroy[good_right_inds]))

            win_l_w_l = 200 #225 #130 
            win_l_w_r = 470+10 #470 #435 #340 # 205
            win_r_w_l = x_current_right - 30
            win_r_w_r = x_current_right + 30
            pts_left = np.array([[win_l_w_l, win_h2], [win_l_w_l, win_h1], [win_l_w_r, win_h1], [win_l_w_r, win_h2]], np.int32)
            cv2.polylines(out_img, [pts_left], False, (0,255,0), 1)
            pts_right = np.array([[win_r_w_l, win_h2], [win_r_w_l, win_h1], [win_r_w_r, win_h1], [win_r_w_r, win_h2]], np.int32)
            cv2.polylines(out_img, [pts_right], False, (255,0,0), 1)

            pts_catch = np.array([[0, circle_height], [width, circle_height]], np.int32)
            cv2.polylines(out_img, [pts_catch], False, (0,120,120), 1)

            for i in range(len(good_right_inds)):
                img = cv2.circle(out_img, (nonzerox[good_right_inds[i]], nonzeroy[good_right_inds[i]]), 1, (255,0,0), -1)

            for window in range(0, nwindows):
                win_y_low = y_current_right - (window + 1) * window_height
                win_y_high = y_current_right - (window) * window_height
                win_x_low = x_current_right - margin
                win_x_high = x_current_right + margin
            
                cv2.rectangle(out_img, (win_x_low - int(width * road_width), win_y_low), (win_x_high - int(width * road_width), win_y_high), (0, 255, 0), 1)
                cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (255, 0, 0), 1)
                
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

            
                if len(good_right_inds) > minpix:
                    x_current_right = int(np.mean(nonzerox[good_right_inds]))

                elif len(nonzeroy[right_lane_inds]) > 2 and len(nonzerox[right_lane_inds]) > 2:
                    p_right = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2) 
                    x_current_right = int(np.polyval(p_right, win_y_high))

                else:
                    pass
           

                if circle_height - 10 <= win_y_low < circle_height + 10:
                    x_location = x_current_right - int(width * road_width/2) 
                    
                    x_location = self.protected_point(x_location)

                    cv2.circle(out_img, (x_location, circle_height), 10, (255, 0, 0), -1)
                
                right_lane_inds.extend(good_right_inds)
        
        else:

            win_l_w_l = 200 #225 #130 
            win_l_w_r = 470 + 10#470 #435 #340 # 205
            win_r_w_l = 1280-420-60#1000 # 1280 - 340
            win_r_w_r = 1280-150#1210 # 1280 - 130 # 550
            pts_left = np.array([[win_l_w_l, win_h2], [win_l_w_l, win_h1], [win_l_w_r, win_h1], [win_l_w_r, win_h2]], np.int32)
            cv2.polylines(out_img, [pts_left], False, (0,255,0), 1)
            pts_right = np.array([[win_r_w_l, win_h2], [win_r_w_l, win_h1], [win_r_w_r, win_h1], [win_r_w_r, win_h2]], np.int32)
            cv2.polylines(out_img, [pts_right], False, (255,0,0), 1)

            pts_catch = np.array([[0, circle_height], [width, circle_height]], np.int32)
            cv2.polylines(out_img, [pts_catch], False, (0,120,120), 1)



            # x_location = 640    
            # cv2.circle(out_img, (x_location, circle_height), 10, (255, 255, 255), -1)
            x_location = self.x_previous
            cv2.circle(out_img, (x_location, circle_height), 10, (255, 255, 255), -1)

        # if x_location == 640:
            

        self.x_previous = x_location

        return out_img, x_location, self.current_line
