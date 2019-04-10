#!/usr/bin/env python
from __future__ import print_function

import roslib
import glob
import os
import sys
import rospy
import cv2
import numpy as np

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic", Image, queue_size=2)
    self.bridge = CvBridge()
    print('Publisher Init complited.')

def main(args):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    imgs_path = os.path.join(cur_dir, 'test_images') + '/*'
    img_list = glob.glob(imgs_path)
    img_index = 0
    img_len = len(img_list)

    rospy.init_node('image_converter_pub', anonymous=True)
    ic = image_converter()
    rate = rospy.Rate(5)

    while not rospy.is_shutdown():
        img_index = 0 if img_index >= img_len-1 else img_index+1
        cur_path = img_list[img_index]
        cv_image = cv2.imread(cur_path)

        print('Image was published.' + cur_path)
        try:
          ic.image_pub.publish(ic.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
          print(e)
        #cv_image = cv2.resize(cv_image, (480, 360))
        #cv2.imshow("Image window", cv_image)
        #cv2.waitKey(3)
        rate.sleep()

if __name__ == '__main__':
    main(sys.argv)

