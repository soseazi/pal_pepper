#!/usr/bin/env python  
import qi
import sys
import time
import numpy as np
import rospy
from geometry_msgs.msg import Twist
import atexit

class cmdvel:

	def __init__(self,ip,port):

		atexit.register(self.__del__)
		#Create NaoQI session
		self.ip = ip ; self.port = port
		self.session = qi.Session()

		try:
			self.session.connect("tcp://" + str(self.ip) + ":" + str(self.port))
		except RuntimeError:
			print ("Connection Error!")
			sys.exit(1)
		self.motion =  self.session.service("ALMotion")

		rospy.init_node('pepper_cmdvel_pub')
		cmd_vel_sub = rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_callback) 
		rospy.spin()

	def cmd_vel_callback(self,msg):
		mod_msg = msg
		x = msg.linear.x
		y = msg.linear.y
		w = msg.angular.z
		self.set_velocity(x,y,z)

	def set_velocity(self,x,y,theta): #m/sec, rad/sec
		#set robot's velocity (note that you should set velocity to (0,0,0) after movement)
		self.motion.move(x,y,theta)

	def __del__(self):		
		self.set_velocity(0.,0.,0.)
		print 'termination with stopping'


if __name__=="__main__":
	c = cmdvel(sys.argv[1],sys.argv[2])

