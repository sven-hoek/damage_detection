#!/usr/bin/env python
from __future__ import print_function

import rospy
import math
from damage_detection.msg import road_damage, road_damage_list
import prius_msgs.msg

class Simple_control:
    def __init__(self):
        self.damage_sub = rospy.Subscriber("boxes_topic", road_damage_list, self.callback)
        self.control_pub = rospy.Publisher("prius", prius_msgs.msg.Control, queue_size=4)

    def callback(self, data):
        n_damages = len(data)
        command = Control()
        command.header = message.header
        command.throttle = 1.0 - 1/3 * n_damages
        command.brake = 0.0
        command.shift_gears = Control.FORWARD

        self.control_pub.publish(command)

if __name__ == '__main__':
    rospy.init_node('simple_control')
    t = Simple_control()
    rospy.spin()
