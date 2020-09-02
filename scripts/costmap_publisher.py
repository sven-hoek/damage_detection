#!/usr/bin/env python

import roslib
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, MapMetaData
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point32, Pose
from std_msgs.msg import Int8


class ObstaclePublisher():
    def __init__(self):
        self.map_pose = Pose()

        self.map_width = 600
        self.map_height = 600
        self.map = OccupancyGrid()
        self.map.header.frame_id = 'map'
        self.map.info.resolution = 1.0
        self.map.info.width = self.map_width
        self.map.info.height = self.map_height
        self.map.data = range(self.map_width * self.map_height)
        grid = np.ndarray((self.map_width, self.map_height), buffer=np.zeros((self.map_width, self.map_height), dtype=np.int8),
    	         dtype=np.int8)
        grid.fill(np.int8(0))
        grid[3:7, 3:7].fill(np.int8(100))
        for i in range(self.map_width*self.map_height):
    		self.map.data[i] = grid.flat[i]

        self.global_costmap = OccupancyGrid()
        self.global_costmap.info.width = 10
        self.global_costmap.info.height = 10
        self.global_costmap.info.resolution = 100
        self.global_costmap.data = range(100)
        grid = np.ndarray((10, 10), buffer=np.zeros((10, 10), dtype=np.int8), dtype=np.int8)
        grid.fill(np.int8(0))
        for i in range(100):
    		self.map.data[i] = grid.flat[i]
        self.global_costmap.info.origin.position.x = -500
        self.global_costmap.info.origin.position.y = -500
        self.global_costmap.header.frame_id = 'map'

        self.obstacle_array = ObstacleArrayMsg()
        obstacle = ObstacleMsg()
        obstacle.polygon.points = [Point32(3.0, 3.0, 0.0),
                                   Point32(3.0, 6.0, 0.0),
                                   Point32(6.0, 6.0, 0.0),
                                   Point32(6.0, 3.0, 0.0)]
        obstacle.radius = 2.5
        self.obstacle_array.obstacles.append(obstacle)

        self.odometry_sub = rospy.Subscriber("base_pose_ground_truth", Odometry, self.odom_callback)
        self.map_pub = rospy.Publisher('move_base/local_costmap/costmap', OccupancyGrid, queue_size=5)
        self.global_costmap_pub = rospy.Publisher('move_base/global_costmap/costmap', OccupancyGrid, queue_size=5)
        self.obstacle_pub = rospy.Publisher('move_base/TebLocalPlannerROS/obstacles', ObstacleArrayMsg, queue_size=5)

    def odom_callback(self, data):
        self.map_pose = data.pose.pose
        #self.map_pose.position.z = 0.0
        self.map_pose.position.x -= 0.5 * (self.map.info.width * self.map.info.resolution)
        self.map_pose.position.y += -0.5 * (self.map.info.height * self.map.info.resolution)

    def publish(self):
        self.map.info.map_load_time = rospy.Time.now()
        self.map.info.origin = self.map_pose
        self.global_costmap.info.origin.position.z = self.map.info.origin.position.z
        self.map_pub.publish(self.map)
        self.global_costmap_pub.publish(self.global_costmap)
        self.obstacle_pub.publish(self.obstacle_array)

if __name__ == '__main__':
    rospy.init_node('costmap_publisher', anonymous=True)
    rate = rospy.Rate(15)
    obstacle_publisher = ObstaclePublisher()
    while not rospy.is_shutdown():
        obstacle_publisher.publish()
        rate.sleep()
