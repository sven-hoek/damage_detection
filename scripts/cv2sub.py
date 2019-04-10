#!/usr/bin/env python
from __future__ import print_function

import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, 'models/research'))
sys.path.insert(0, os.path.join(cur_dir, 'models/research/slim'))
sys.path.insert(0, os.path.join(cur_dir, 'models/research/object_detection'))


import roslib
import rospy
import numpy.core.multiarray
import cv2
import numpy as np
import tensorflow as tf
from damage_detection.msg import road_damage, road_damage_list

import custom_model as cm

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def create_holelist(nn_list):
    holes = road_damage_list()
    for x0, y0, x1, y1, prob in nn_list:
        hole = road_damage()
        hole.top_left.x = x0
        hole.top_left.y = y0
        hole.bottom_right.x = x1
        hole.bottom_right.y = y1
	hole.probability = prob
	holes.damages.append(hole)
    return holes


class image_converter:
  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("prius/front_camera/image_raw", Image, self.callback)
    self.damage_pub = rospy.Publisher("boxes_topic", road_damage_list, queue_size=16)
    print('Subscriber Init complited.')

  def callback(self,data):
    try:
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    hole_list = create_holelist(cm.get_objects(cv_image))
    try:
        self.damage_pub.publish(hole_list)
    except CvBridgeError as e:
        print(e)

    # cv_image = cv2.resize(cv_image, (480, 360))
    # cv2.imshow("Image window", cv_image)
    # cv2.waitKey(3)



def main(args):

    rospy.init_node('image_converter_sub', anonymous=True)
    ic = image_converter()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
