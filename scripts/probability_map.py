#!/usr/bin/env python

import numpy as np
import cv2
import rospy
import time

from damage_detection.msg import road_damage, road_damage_list
from nav_msgs.msg import OccupancyGrid, Odometry

DEBUG_MODE = True

WIDTH = 800
HEIGHT = 800

D2P_TRANSFORM = 59.045

K = 0.9

X = 0.0
Y = 0.0
Theta = 0.0

X_prev = 0.0
Y_prev = 0.0
Theta_prev = 0.0

dX = 0.0
dY = 0.0
dTheta = 0.0

def createMap(w, h):
    return np.zeros((w, h), dtype=np.uint8)


def distance2pixels(a):
    return a * D2P_TRANSFORM


def intersectMaps(mask): # make intersection of view areas after given dX, dY, dTheta
    global dX
    global dY
    global dTheta

    x_lt = distance2pixels(-dX)
    y_lt = distance2pixels(-dY)

    x_rt =  np.cos(-dTheta) * (WIDTH - 1) + x_lt
    y_rt = -np.sin(-dTheta) * (HEIGHT - 1) + y_lt

    x_lb = np.sin(-dTheta) * (WIDTH - 1) + x_lt
    y_lb = np.cos(-dTheta) * (HEIGHT - 1) + y_lt

    src_points = np.float32([[0, 0], [(WIDTH - 1), 0], [0, (HEIGHT - 1)]])
    dst_points = np.float32([[x_lt, y_lt], [x_rt, y_rt], [x_lb, y_lb]])

    affine_matrix = cv2.getAffineTransform(src_points, dst_points)

    print('mask', type(mask))
    print('affine', affine_matrix)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    print('mask', type(mask))

    transformed_image = cv2.warpAffine(mask, affine_matrix, (WIDTH, HEIGHT))

    return transformed_image


def updateMap(new, mask):
    h = createMap(WIDTH, HEIGHT)
    mask = mask[:, :, 0]
    h = new*(1 - K) + mask*K

    return h


def cordinates_to_image(hole_list, image_shape):
    image = np.zeros(image_shape, dtype=np.uint8)
    print(hole_list)
    for hole in hole_list.damages:
        image = cv2.rectangle(image, (int(hole.top_left.x), int(hole.top_left.y)),\
                             (int(hole.bottom_right.x), int(hole.bottom_right.y)), int(hole.probability*255), -1)
    return image


def bird_eye_transform(original_image):
    src = np.float32([[310, 390], [475, 390], [-800, 430 + 130], [800 + 635, 450 + 130]])
    dst = np.float32([[0, 0], [800, 0], [0, 800], [800, 800]])
    m = cv2.getPerspectiveTransform(src, dst)
    image = cv2.warpPerspective(original_image, m, (800, 800))
    return image


class image_converter:
    def __init__(self):
        self.m = createMap(WIDTH, HEIGHT)
        self.p = createMap(WIDTH, HEIGHT)

        self.map_msg = OccupancyGrid()
        self.map_msg.info.width = WIDTH
        self.map_msg.info.height = HEIGHT
        self.map_msg.info.resolution = 0.2
        self.map_msg.data = range(WIDTH*HEIGHT)
        self.image_sub = rospy.Subscriber("boxes_topic", road_damage_list, self.callback, queue_size = 1)
        self.carpos_sub = rospy.Subscriber("base_pose_ground_truth", Odometry, self.get_position)
        self.prob_pub = rospy.Publisher("probability_topic", OccupancyGrid, queue_size=16)

    if DEBUG_MODE:
            print('Probability Init complited.')

    def callback(self, data):
        t = time.time()
        self.get_deltas()

        print('Coord shift ', dX, dY, dTheta)

        image = cordinates_to_image(data, (WIDTH, HEIGHT))
        bird_image = bird_eye_transform(image)

        if DEBUG_MODE:
            print('Made bird view transform')

        print(intersectMaps(bird_image).shape)
        self.p = intersectMaps(bird_image)
        #self.p = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

        if DEBUG_MODE:
            print('Maps intersected')

        self.m = updateMap(self.m, self.p)
        if DEBUG_MODE:
            print('Map updated')

        self.map_msg.header.stamp = rospy.Time.now()

        if DEBUG_MODE:
            print('Start reshaping...')

        x = cv2.resize(self.m, (400, 400))
        cv2.imshow('1', x)
        t1 = time.time()
        self.map_msg.data[:] = x.reshape(-1) #self.m.reshape(-1)
        t2 = time.time()
        self.prob_pub.publish(self.map_msg)
        t3 = time.time()
        print()
        print('T_RESHAPE: ', t2-t1)
        print('T_PUBLISH: ', t3-t2)
        print()
        #print('LOOK: ', self.map_msg)

        if DEBUG_MODE:
            print('End reshaping')
            cv2.imshow('ProbMap', self.m)
            print('fps: ', 1/(time.time() - t))
            cv2.waitKey(1)

    #print(np.amax( self.map_msg))

    def get_deltas(self):
        global Y
        global Theta
        global X_prev
        global Y_prev
        global Theta_prev
        global dX
        global dY
        global dTheta

        dX = X - X_prev
        dY = Y - Y_prev
        dTheta = Theta - Theta_prev

        X_prev = X
        Y_prev = Y
        Theta_prev = Theta

    def get_position(self, data):
        global X
        global Y
        global Theta

        X = data.pose.pose.position.x
        Y = data.pose.pose.position.y
        Theta = data.pose.pose.position.z
        #print(X, Y, Theta, dX, dY, dTheta)


if __name__ == '__main__':
    print(cv2.__version__)
    rospy.init_node('image_probability_sub', anonymous=True)
    ic = image_converter()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
