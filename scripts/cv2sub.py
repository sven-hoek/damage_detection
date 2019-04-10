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
import cv2
import numpy as np
import tensorflow as tf
import damage_detection.msg

import custom_model as cm

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:
  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("image_topic", Image, self.callback)
    self.image_pub = rospy.Publisher("boxes_topic", String, queue_size=16)
    print('Subscriber Init complited.')
    
  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    cv_image = cm.draw_rect(cv_image)
    cv_image = cv2.resize(cv_image, (480, 360))
    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

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

