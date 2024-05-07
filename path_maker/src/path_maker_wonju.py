#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import os
import sys
import rospy
import rospkg
from pyproj import Proj, transform
from sensor_msgs.msg import NavSatFix
from morai_msgs.msg  import EgoVehicleStatus, GPSMessage
from math import pi, sqrt
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
import tf

warnings.simplefilter(action='ignore', category=FutureWarning)

class PathMaker :
    def __init__(self):
        rospy.init_node('path_maker', anonymous=True)
    
        self.cnt = 0

        arg = rospy.myargv(argv = sys.argv)
        self.path_folder_name = arg[1]
        self.make_path_name = arg[2]
        
        self.longitude = 0
        self.latitude = 0
        self.altitude = 0

        rospy.Subscriber("/gps",GPSMessage, self.gps_callback)

        self.prev_longitude = 0
        self.prev_latitude = 0

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('path_maker')
        full_path = pkg_path +'/'+ self.path_folder_name+'/'+self.make_path_name+'.txt'
        self.f = open(full_path, 'w')

        rate = rospy.Rate(30) 
        while not rospy.is_shutdown():
            self.path_make()
            rate.sleep()    

        self.f.close()
        

    def path_make(self):
        utmk_coordinate = Point() 

        self.proj_UTM = Proj(proj='utm', zone = 52, elips='WGS84', preserve_units=False)

        xy_zone = self.proj_UTM(self.longitude, self.latitude)
        self.x, self.y = xy_zone[0], xy_zone[1]

        utmk_coordinate.x = self.x - 402300.0
        utmk_coordinate.y = self.y - 4132900.0
        utmk_coordinate.z = 0

        distance = sqrt(pow(utmk_coordinate.x - self.prev_longitude, 2) + pow(utmk_coordinate.y - self.prev_latitude, 2))

        # mode = 0
        if distance > 0.3:   #0.3
            data='{0}\t{1}\n'.format(utmk_coordinate.x, utmk_coordinate.y)
            self.f.write(data)
            self.cnt += 1
            self.prev_longitude = utmk_coordinate.x
            self.prev_latitude = utmk_coordinate.y
        
            print(self.cnt, utmk_coordinate.x, utmk_coordinate.y)

    def gps_callback(self, msg): 
        self.longitude = msg.longitude
        self.latitude = msg.latitude
        self.altitude = msg.altitude

        
if __name__ == '__main__':
    try:
        path_maker_=PathMaker()
    except rospy.ROSInterruptException:
        pass

