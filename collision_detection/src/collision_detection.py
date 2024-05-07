#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from morai_msgs.msg import CollisionData

class CollisionDetection:
    def __init__(self):
        rospy.init_node('collision_detection', anonymous=True)
        rospy.Subscriber("/CollisionData", CollisionData, self.collisionCB) ## Vehicle Status Subscriber

        rate = rospy.Rate(30) 
        self.collision_cnt = 0              
        while not rospy.is_shutdown():
            rate.sleep()
            

    def collisionCB(self, msg):
        if msg.collision_object != []:
            self.collision_cnt += 1
            print("Collision Detected:", self.collision_cnt)

if __name__ == '__main__':
    try:
        collision_detection= CollisionDetection()
    except rospy.ROSInterruptException:
        pass