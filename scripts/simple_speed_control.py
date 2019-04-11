#!/usr/bin/env python
from __future__ import print_function

import rospy
import math
from damage_detection.msg import road_damage, road_damage_list
from nav_msgs.msg import Odometry
import prius_msgs.msg

class Simple_control:
    def __init__(self):
        #self.damage_sub = rospy.Subscriber("boxes_topic", road_damage_list, self.callback)
        self.odometry_sub = rospy.Subscriber("base_pose_ground_truth", Odometry, self.odom_callback)
        self.control_pub = rospy.Publisher("prius", prius_msgs.msg.Control, queue_size=4)
        self.odometry = Odometry()
        self.linear_velocity = 0.0
        self.desired_speed = 10.0

    def callback(self, data):
        n_damages = len(data.damages)
        command = prius_msgs.msg.Control()
        if n_damages > 0:
            command.throttle = 0.0
            command.brake = 1.0
        elif self.linear_velocity < 10:
            command.throttle = 1.0
            command.brake = 0.0
        else:
            command.throttle = 0.0
            command.brake = 0.0
        command.shift_gears = prius_msgs.msg.Control.FORWARD
        self.control_pub.publish(command)

    def odom_callback(self, data):
        self.odometry = data
        linvel = data.twist.twist.linear
        self.linear_velocity = math.sqrt(linvel.x**2 + linvel.y**2)
        print("Current speed {}m/s".format(self.linear_velocity))

        difference = self.linear_velocity - self.desired_speed
        command = prius_msgs.msg.Control()
        if difference > 0:
            # actual speed larger than desired speed -> brake
            command.throttle = 0.0
            command.brake = difference
        else:
            command.throttle = -difference
            command.brake = 0.0
        command.shift_gears = prius_msgs.msg.Control.FORWARD
        self.control_pub.publish(command)

if __name__ == '__main__':
    rospy.init_node('simple_control')
    t = Simple_control()
    rospy.spin()
