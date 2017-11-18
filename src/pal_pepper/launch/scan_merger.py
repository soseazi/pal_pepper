#!/usr/bin/env python  
import qi
import sys
import time
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import LaserScan

class scan_merger:
	def __init__(self):
		self.s1 = rospy.Subscriber('/pepper_robot/laser', LaserScan, self.cb1, queue_size=1)
		self.s2 = rospy.Subscriber('/scan', LaserScan, self.cb2, queue_size=1)	
		self.p = rospy.Publisher('/scan_merged',LaserScan,queue_size=1)
		self.msg = LaserScan()
		self.msg.header.frame_id = 'base_footprint'
		self.msg.angle_min = -np.pi
		self.msg.angle_max = np.pi
		self.msg.angle_increment = 1./np.pi
		self.msg.ranges = -1*np.ones(360)

	def cb1(self,msg):
		self.p.publish(msg)

	def cb2(self,msg):
	
		self.p.publish(msg)

def main(msg):	
	global motion

	motion.setStiffnesses("Head",1.0)
	motion.setAngles("Head",[0.,0.],0.05)
	#print motion.getAngles("Head",False)
	time.sleep(2)




if __name__=="__main__":
	rospy.init_node('scan_merger')
	sm = scan_merger()
	rospy.spin()

