#!/usr/bin/env python  
import qi
import sys
import time
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo


def main(msg):	
	global motion

	motion.setStiffnesses("Head",1.0)
	motion.setAngles("Head",[0.,0.],0.05)
	#print motion.getAngles("Head",False)
	time.sleep(2)


if __name__=="__main__":
	ip = sys.argv[1]
	port = sys.argv[2]
	session = qi.Session()

	try:
		session.connect("tcp://" + str(ip) + ":" + str(port))
	except RuntimeError:
		print ("Connection Error!")
		sys.exit(1)

	motion =  session.service("ALMotion")
	autolife = session.service("ALAutonomousLife")
	autolife.setState('disabled')
	posture = session.service("ALRobotPosture")
	posture.goToPosture("Stand",0.3)
	
	rospy.init_node('head_fix')
	rospy.Subscriber('/pepper_robot/camera/depth/camera_info', CameraInfo, main, queue_size=1)
	rospy.spin()

