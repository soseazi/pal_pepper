#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

#TODO time limit in get dobjects functions

'''
Please refer for any publication
Beom-Jin, Lee, et al. "Perception-Action-Learning System for Mobile Social-Service Robots using Deep Learning." AAAI. 2018

Beom-Jin Lee, Jinyoung Choi, Chung-Yeon Lee, Kyung-Wha Park, Sungjun Choi, Cheolho Han,
Dong-Sig Han, Christina Baek, Patrick Emaase, Byoung-Tak Zhang

School of Computer Science and Engineering, Cognitive Science Program, Interdisciplinary Program in Neuroscience
Seoul National University, Seoul, Korea, Republic of
'''

import argparse
import sys
import time
import cv2
import numpy as np
import signal
from threading import Thread
import thread
import os
import atexit
from math import radians, degrees
import json
import datetime
import shutil
import subprocess
import signal


#external process module
import pexpect

#speech recognition modulesshpass -p 1847! ssh nao@192.168.1.X "rm /home/nao/record/recog.wav"
import io
from google.cloud import speech

#ROS modules
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion , Twist, Pose, PoseStamped, Vector3
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image,CameraInfo
from std_msgs.msg import Int32,String,ColorRGBA
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid, Odometry
from tf import TransformListener, Transformer, transformations
from visualization_msgs.msg import Marker
from actionlib_msgs.msg import GoalID

from std_srvs.srv import Empty
from pal_pepper.msg import objs, objs_array

import dynamic_reconfigure.client


#Pepper naoqi modules
import qi
import pepper_config


def exit_handler():
	print 'exit'

class pepper_io():

	# Robot Name
	robot_name = "pepper"

	def __init__(self,params): #requires ip and port
		self.params = params
		self.person_names = params['person_names']
		rospy.init_node("pepper_io")

		#Create NaoQI session
		atexit.register(self.__del__)
		self.ip = params['ip'] ; self.port = params['port']
		self.session = qi.Session()

		try:
			self.session.connect("tcp://" + self.ip + ":" + str(self.port))
		except RuntimeError:
			print ("Connection Error!")
			sys.exit(1)


		# Camera
		self.video = self.session.service("ALVideoDevice")
		video_subs_list = ['rgb_t_0','rgb_b_0','dep_0']
		print self.video.getSubscribers()
		for sub_name in video_subs_list :
			print sub_name, self.video.unsubscribe(sub_name)


		self.rgb_top = self.video.subscribeCamera('rgb_t',0,2,11,20)   #name, idx, resolution, colorspace, fps
		self.rgb_bottom = self.video.subscribeCamera('rgb_b',1,2,11,20)
		offset_x = -120
		offset_y = -160 + 30
		self.x_temp = -(np.tile(np.arange(240).reshape(240,1) , (1,320)).reshape(240,320,1) + offset_x)
		self.y_temp = -(np.tile(np.arange(320).reshape(1,320) , (240,1)).reshape(240,320,1) + offset_y)
		self.depth = self.video.subscribeCamera('dep',2,1,17,20)
		print self.video.getSubscribers()

		#NAOqi memory
		self.ALMEM = self.session.service("ALMemory")

		# sound localization
		self.sound_loc = self.session.service("ALSoundLocalization")
		self.sound_loc_subsub=self.sound_loc.subscribe('SL')
		self.sound_loc_sub = self.ALMEM.subscriber("ALSoundLocalization/SoundLocated")
		self.sound_loc_sub.signal.connect(self.callback_sound_loc)


		#Audio record
		self.record_delay = 2
		self.record_time = 0
		self.recording = False

		self.audio_player = self.session.service("ALAudioPlayer")
		self.audio_recorder = self.session.service("ALAudioRecorder")
		self.audio_recorder.stopMicrophonesRecording()

		self.thread_recording = Thread(target=self.record_audio,args=(None,))
		self.thread_recording.daemon = True
		self.audio_terminate = False

		#GoogleSpeech
		self.sr_flag = False
		self.speech_memory = ""
		#Important: to use Google speech, need account (https://cloud.google.com/speech/docs/auth)
		#unindent --> self.speech_client = speech.Client.from_service_account_json('speech_auth/xxx.json')
		self.speech_hints = []

		# animation & posture
		self.animation = self.session.service("ALAnimationPlayer")
		self.posture   = self.session.service("ALRobotPosture")
		self.tracker   = self.session.service("ALTracker")
		self.tracker.registerTarget('Face', 0.1)


		# navigation & mapping
		self.motion =  self.session.service("ALMotion")
		self.nav_qi = self.session.service("ALNavigation")
		self.nav_as = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
		self.cmd_vel_pub = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
		self.goal_cancel_pub = rospy.Publisher('/move_base/cancel',GoalID,queue_size=1)
		self.nav_as.wait_for_server()
		self.static_map_service = dynamic_reconfigure.client.Client('/move_base/global_costmap/static_layer')
		self.map = None
		print 'connected to /move_base'

		self.transform = TransformListener()
		self.transformer = Transformer(True,rospy.Duration(10.0))
		self.sub_map = rospy.Subscriber('/move_base/global_costmap/costmap', OccupancyGrid, self.callback_map, queue_size=1)
		self.waypoints = {}

		# Text to speech
		self.tts = self.session.service("ALTextToSpeech")
		self.tts.setParameter("defaultVoiceSpeed", 100)
		self.tts.setVolume(1.0) #TODO change in real competition

		#CVBridge for image publishing
		self.cvbridge = CvBridge()

		#global localization
		self.localization_srv = rospy.ServiceProxy('global_localization',Empty)
		self.pose_cov_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.callback_pose, queue_size=1)
		self.global_localization_timelimit = 60
		self.global_localization_cov_threshold = 0.01
		self.initial_pose_pub = rospy.Publisher('/initialpose' ,PoseWithCovarianceStamped,queue_size=1)

		#clear costmap
		self.map_clear_srv = rospy.ServiceProxy('move_base/clear_costmaps',Empty)
		self.map_clear_srv()

		#Collistion Avoidance
		self.coll_sub = self.ALMEM.subscriber("ALMotion/MoveFailed")
		self.coll_sub.signal.connect(self.callback_obstacle)
		self.ortho_sec_dist = 0.05
		self.tan_sec_dist = 0.05
		self.motion.setOrthogonalSecurityDistance(self.ortho_sec_dist)#0.05)
		self.motion.setTangentialSecurityDistance(self.tan_sec_dist)#0.05)
		self.reflex_going = False

		self.touch_stat = []
		self.touch_srv = self.session.service('ALTouch')
		self.touch_mem_sub = self.ALMEM.subscriber("TouchChanged")
		self.touch_mem_sub_callback = self.touch_mem_sub.signal.connect(self.callback_touch)

		#Perception
		self.objs_history = []
		self.objs_history_idx = []

		self.objs_history_max = 10

		self.reid_history = []
		self.reid_history_idx = []

		self.reid_history_max = 3

		self.pose_history = []
		self.pose_history_idx = []

		self.pose_history_max = 3

		self.cap_history = []
		self.cap_history_idx = []

		self.cap_history_max = 3

		self.objects = objs_array()
		self.people_waving = objs_array()
		self.people_identified = objs_array()


		self.reid_targets = objs_array()

		self.sub_obj = rospy.Subscriber(self.params['obj_topic'], objs_array, self.callback_objs, queue_size=1)
		self.sub_pose = rospy.Subscriber(self.params['pose_topic'], objs_array, self.callback_waving, queue_size=1)
		self.sub_identify = rospy.Subscriber(self.params['reid_topic'], objs_array, self.callback_identify, queue_size=1)
		self.sub_cap = rospy.Subscriber(self.params['captioning_topic'], objs_array, self.callback_caps, queue_size=1)

		self.reid_target_pub = rospy.Publisher(self.params['reid_target_topic'] ,objs_array,queue_size=1)

		
		
		self.perception = objs_array()
		#self.thread_matching = Thread(target=self.match_perception,args=(None,))
		#self.thread_matching.daemon = True
		#self.thread_matching.start()

		#show integrated perception
		if self.params['show_integrated_perception'] :
			cv2.startWindowThread()
			cv2.namedWindow('PIO_perception')
			self.thread_show = Thread(target=self.show_perception,args=(None,))
			self.thread_show.daemon = True
			self.thread_show.start()

		#for captioning
		self.ext_process_modules=[]
		self.pub_cap_req = rospy.Publisher('captioning_request',objs_array,queue_size=1)
		self.sub_caption = rospy.Subscriber("captioning_result", objs_array, self.callback_captioning)
		self.captioning_req_time = time.time()
		self.captioning_flag = False
		self.captioning_result = []
		
		#for VQA
		self.pub_vqa_req = rospy.Publisher('vqa_request',objs_array,queue_size=1)
		self.sub_vqa = rospy.Subscriber("vqa_response", objs_array, self.callback_vqa)
		self.vqa_req_time = time.time()
		self.vqa_flag = False
		self.vqa_result = []

		#barcode
		self.barcode_service = self.session.service("ALBarcodeReader")
		self.barcode_service.subscribe("pio_barcode")
		self.barcode_sub = self.ALMEM.subscriber("BarcodeReader/BarcodeDetected")
		self.barcode_sub.signal.connect(self.callback_barcode)

		#data recording
		self.data_recording = False
		self.data_recording_dir = ''
		self.muted = False
		if 'muted' in self.params:
			self.muted = self.params['muted']

		self.clear_reid_targets()
	
		
		#concept map
		self.concept_map = (np.ones((480,640))*255).astype('uint8')
		self.annotated_scene = (np.ones((480,640))*255).astype('uint8')
		self.img_concepts = []
		self.text_concepts = []
		self.obj_concepts = []
		
		self.obj_corpus = []
		self.obj_corpus_img = []
		
		ld = os.listdir('./object_images/')
		
		for fn in ld :
			if fn.split('.')[-1] != 'png' or len(fn.split('_'))!=3 : continue
			on = fn.split('.')[0].split('_')[1]
			self.obj_corpus.append(on)
			self.obj_corpus_img.append( cv2.imread(fn) )

			
		self.obj_corpus = list(set(self.obj_corpus))
		self.obj_concept_data = np.zeros((1,len(self.obj_corpus)))
		
		self.text_corpus = []
		self.text_concept_data = np.zeros((1,1))
		
		self.concept_data = np.zeros((1,1))	
		
		self.enable_concept_map = False
		
		
		self.set_static_map(True)
		self.beep_volume = 70 #(0~100)
		
		self.enable_pororo = False

		print os.getpid()
		print 'PIO ready'
		self.say("ready")

	
	def callback_touch(self,msg):
		self.touch_stat = msg
	
	def __del__(self):
		#self.asr_qi.unsubscribe("Test_ASR")
		print 'Stop audio recording...'
		self.audio_recorder.stopMicrophonesRecording()
		print 'Removing audio record files from robot...'
		cmd = 'sshpass -p 1847! ssh nao@'+str(self.ip)+' "rm /home/nao/record/recog.wav"'
		os.system(cmd)
		print cmd
		print 'Unsubscribing cameras...'
		video_subs_list = ['rgb_t_0','rgb_t_0','dep_0','pc_0']
		for sub_name in video_subs_list :
			print sub_name, self.video.unsubscribe(sub_name)
		print 'Stop the wheels...'
		self.set_velocity(0.,0.,0.)
		if self.data_recording :
			print 'Reindexing rosbag'
			cmd = 'rosbag reindex ' + self.data_recording_dir + '/rosbag/*.bag.active'
			os.system(cmd)

		print 'Killing all external processes...'
		for ext_proc in self.ext_process_modules:
			ext_proc.send('\003')
			ext_proc.close(force=True)
			time.sleep(1)
			if ext_proc.isalive() :
				print 'error!!!! - cannot kill child process'
		os.kill(os.getpid(), 9)
		print 'pepper_io is safely deleted!'
		
		
	def set_initial_pose(self,p,o,cov1=0.25,cov2=-0.006):
		msg = PoseWithCovarianceStamped()
		msg.header.frame_id = "map";
		
		msg.pose.pose.position.x = p[0];
		msg.pose.pose.position.y = p[1];
		msg.pose.pose.position.z = p[2];

		msg.pose.pose.orientation.x = o[0];
		msg.pose.pose.orientation.y = o[1];
		msg.pose.pose.orientation.z = o[2];		
		msg.pose.pose.orientation.w = o[3];		
	
		for i in range( len(msg.pose.covariance) ) : msg.pose.covariance[i] = 0
		msg.pose.covariance[0] = cov1
		msg.pose.covariance[1] = cov2
		msg.pose.covariance[6] = cov2
		msg.pose.covariance[7] = cov1
		msg.pose.covariance[21] = cov2	
		self.initial_pose_pub.publish(msg)
		
	def set_initial_pose_wp(self,wp,cov1=0.25,cov2=-0.006):
		
		w = self.waypoints[wp]
		p = np.array( [w[0] , w[1] , 0.0] )
		o = self.yaw_to_quat( w[2] )
		
		self.set_initial_pose(p,o,cov1,cov2)
		
	def show_perception(self,dummy):
		joint_colors = {
			0 : (0,0,255),
			1 : (0,255,0),
			2 : (255,0,0),
			3 : (0,255,255),
			4 : (255,0,255),
			5 : (255,255,0),
			6 : (255,255,255),
			7 : (0,0,100),
			8 : (0,100,0),
			9 : (100,0,0),
			10 : (0,100,100),
			11 : (100,0,100),
			12 : (100,100,0),
			13 : (100,100,100),
			14 : (0,0,50),
			15 : (0,50,0),
			16 : (50,0,0),
			17 : (50,0,50),
		}

		disp_tags = ['waving','sitting','man','woman', 'white_cloth', 'blue_cloth', 'red_cloth', 'black_cloth', 'green_cloth']

		while not rospy.is_shutdown() :
			tic = time.time()
			per = self.get_perception(reid=True,pose=True,captioning=True,record=False)

			#print '1' , len(per.objects) , per.msg_idx

			if per.msg_idx == 0 : continue
			rgb = self.rosimg_to_numpyimg( per.scene_rgb )
			cloud = self.rosimg_to_numpyimg( per.scene_cloud , 'passthrough')
			cloud = cloud/4.96
			mask = (cloud > 0).astype('int')
			cloud = 1.0 - cloud
			cloud *= mask
			cloud *= 255
			cloud = cloud[:,:,0].reshape((240,320,1)).astype('uint8')
			rgbd = (0.7*rgb + 0.3*cloud).astype('uint8')
			cv2.rectangle(rgbd, (0,0) , (300,15) , (0,0,0) , -1 )

			for o in per.objects :
				x = o.x - o.h
				y = o.y - o.w
				h = 2*o.h ; w = 2*o.w
				cv2.rectangle(rgbd, (y,x) , (y+w,x+h) , (0,255,0) , 2 ) #bounding box
				
				if o.person_name != '' : name = o.person_name
				elif o.sub_class != '' : name = o.sub_class				
				else : name=o.class_string
					
				for t in disp_tags :
					if t in o.tags : name+='('+t+')'

				cv2.rectangle(rgbd, (y-1,x-1) , (y+w+1,x+10) , (0,0,0) , -1 )
				cv2.putText(rgbd,name,(y,x+7),cv2.FONT_HERSHEY_SIMPLEX,0.3,color=(255,255,255),thickness=1)

				if len(o.joints) > 0 :
					for i in range(0,18) :
						if o.joints[2*i] > 0 and o.joints[2*i+1] > 0 :
							cv2.circle(rgbd, (y+int(o.joints[2*i+1]),x+int(o.joints[2*i])) , 2, joint_colors[i] , -1)
							
			self.annotated_scene = cv2.resize(rgbd ,(640,480))
			
			cv2.imshow('PIO_perception',cv2.resize(rgbd ,(640,480)) )
			




	def get_perception(self,fil = None,reid=True,pose=True,captioning=True,time_limit = 3.0,record=True):
		if record : self.record_image()
		objs = self.match_perception(reid,pose,captioning)
		td =  time.time() - objs.header.stamp.secs
		if td > time_limit :
			return objs_array()
		if fil is None : return objs
		else :
			result = objs_array()
			result.header = objs.header
			result.msg_idx = objs.msg_idx
			result.scene_rgb = objs.scene_rgb
			result.scene_cloud = objs.scene_cloud

			for item in objs.objects :
				if item.class_string in fil :
					result.objects.append(item)

			return result

	def match_perception(self,reid=True,pose=True,captioning=True):

		tic = time.time()
		required = 0
		required += int(reid) + int(pose) + int(captioning)

		ooo = self.objs_history[::-1]
		if len(ooo) == 0 : return objs_array()

		rrr = self.reid_history[::-1]
		ppp = self.pose_history[::-1]
		ccc = self.cap_history[::-1]

		oooi = self.objs_history_idx[::-1]
		rrri = self.reid_history_idx[::-1]
		pppi = self.pose_history_idx[::-1]
		ccci = self.cap_history_idx[::-1]
		#print 'o : ' , oooi 
		#print 'r : ' , rrri
		#print 'p : ' , pppi
		#print 'c : ' , ccci
		contingency = None

		for k in range( len( oooi ) ) :
			got = 0
			r,p,c = None,None,None
			if (oooi[k] in rrri and reid) :
				r = rrr[  rrri.index(oooi[k])  ] ; got+=1
			if (oooi[k] in pppi and pose) :
				p = ppp[ pppi.index(oooi[k])  ] ; got+=1
			if (oooi[k] in ccci and captioning) :
				c = ccc[ ccci.index(oooi[k])  ] ; got+=1

			for o in ooo[k].objects :
				if o.class_string != 'person' : continue
				if p is not None :
					if contingency is None : contingency = p
					for pp in p.objects :
						if pp.object_id == o.object_id: 
							o.isWaving = pp.isWaving
							o.isRWaving = pp.isRWaving
							o.isLWaving = pp.isLWaving
							o.isSitting = pp.isSitting
							o.isLying = pp.isLying
							o.isLPointing = pp.isLPointing
							o.isRPointing = pp.isRPointing
							o.joints = pp.joints
							o.tags += pp.tags
							o.pose_wrt_robot = pp.pose_wrt_robot
							o.pose_wrt_odom = pp.pose_wrt_odom
							o.pose_wrt_map = pp.pose_wrt_map

				if r is not None :
					for rr in r.objects :
						if rr.object_id == o.object_id :
							o.person_id = rr.person_id
							o.person_name = rr.person_name
							o.reid_score = rr.reid_score
							o.tags += rr.tags

				if c is not None :
					for cc in c.objects :
						if cc.object_id == o.object_id :
							o.captions = cc.captions
							o.tags += cc.tags

				o.tags = list(set(o.tags))

			if got == required : return ooo[k]
		if contingency is not None : return contingency
		return ooo[ 0  ]


	def set_static_map(self,enabled):
		setting={'enabled':enabled}
		self.static_map_service.update_configuration(setting)

	def save_waypoints(self,filename):
		self.say('saving all waypoints')
		ff = open('waypoints/'+filename,'w')
		for loc in self.waypoints.keys():
			ff.write(loc + ',' + str(  self.waypoints[loc][0]  ) + ',' + str(  self.waypoints[loc][1]  ) + ',' + str(  self.waypoints[loc][2]  ) + '\n')
			ff.flush()
		ff.close()

	def load_waypoints(self,filename,say=True):
		if say : self.say('loading waypoints from file')
		self.waypoints = {}
		ff = open('waypoints/'+filename,'r').readlines()
		for line in ff :
			temp = line.split(',')
			if len(temp) == 4 : self.add_waypoint( temp[0],  [ float(temp[1]),float(temp[2]),float(temp[3]) ]  )

	def save_reid_targets(self):
		self.say('saving all people')
		for target in self.reid_targets.objects :
			name = target.person_name
			idx = target.person_id
			img = self.rosimg_to_numpyimg(target.cropped)
			cv2.imwrite('reid_targets/'+str(idx)+'_'+name+'.png',img)

	def load_reid_targets(self,names=[]):
		self.say('loading people')
		all_target_files = os.listdir('./reid_targets')
		self.reid_targets = objs_array()
		for target in all_target_files :
			if target.split('_')[1].split('.')[0] in names or len(names)==0 :
				idx = int(target.split('_')[0])
				name = target.split('_')[1].split('.')[0]
				img = cv2.imread('reid_targets/'+target)

				self.add_reid_target(idx,name,img=img,person_obj=None)

	def add_waypoint(self,name,location=None):
		if location is None :
			temp = self.get_loc()
			pos = temp[0]
			quat = temp[1]
			rpy=transformations.euler_from_quaternion((quat[0],quat[1],quat[2],quat[3]))
			yaw = rpy[2]
			location = [pos[0],pos[1],yaw]
		self.waypoints[name] = location
		print("Waypoint added ",name,location)


	def go_to_waypoint(self,name,wait=True,clear_costmap=False,wait_timeout = 0):
		if name in self.waypoints.keys() :
			wp = self.waypoints[name]
			return self.go_to_goal(wp[0],wp[1],wp[2],wait,clear_costmap,wait_timeout)
		else :
			self.say('invalid name')
			return 1


	def add_reid_target(self,idx,name,img=None,person_obj=None):
		if img is None and person_obj is None : raise ValueError

		if img is not None :
			person_obj = objs()
			person_obj.cropped = self.numpyimg_to_rosimg(img)
			person_obj.class_string = 'person'

		person_obj.person_id = idx
		person_obj.person_name = name

		self.reid_targets.objects.append(person_obj)
		self.reid_target_pub.publish(self.reid_targets)
		print 'added ',name
		return None

	def clear_reid_targets(self):
		print 'clear reid targets'
		self.reid_targets = objs_array()
		self.reid_target_pub.publish(self.reid_targets)


	def remove_reid_target(self,idx):
		#delete all if idx==-1
		if idx == -1 : del self.reid_targets.objects[:]
		else:
			for i in range(len(self.reid_targets.objects)):
				if self.reid_targets.objects[i].person_id == idx :
					del self.reid_targets.objects[i]
					break
		self.reid_target_pub.publish(self.reid_targets)
		return None

	def get_reid_targets(self):
		result = []
		for item in self.reid_targets.objects :
			result.append( [item.person_id, item.person_name] )
		return result


	def callback_barcode(self,msg):
		if len(msg) == 0 : return None
		print 'Barcode detected : ' ,  msg[0][0]
		self.speech_memory = msg[0][0]

	def callback_map(self,msg):
		self.map = msg
		self.map_time = msg.header.stamp
		self.map_00_pose = np.array(   [msg.info.origin.position.x , msg.info.origin.position.y , 0]   )
		self.map_00_quat = np.array(   [msg.info.origin.orientation.x , msg.info.origin.orientation.y , msg.info.origin.orientation.z , msg.info.origin.orientation.w]   )
		self.map_res = msg.info.resolution
		self.map_h = msg.info.height
		self.map_w = msg.info.width
		self.map_img = np.asarray(msg.data, dtype='int8')
		self.map_img = self.map_img.reshape((self.map_h,self.map_w))
		
		 #raise ValueError
	def is_occupied(self,x,y,size=0.25,allow_unknown=True): #in meters		
	
		if self.map is None : return False
		#reverse x,y in map image
		x_map = y - self.map_00_pose[0]
		y_map = x - self.map_00_pose[1]

		x_map = int( x_map / self.map_res )
		y_map = int( y_map / self.map_res )
		size_map = max(1, int( size/self.map_res ))
		
		max_val = np.amax( self.map_img[x_map-size_map:x_map+size_map,y_map-size_map:y_map+size_map] )
				
		
		if max_val > 10 : result = True
		else : 
			if max_val == -1 and not allow_unknown:
				result = True
			else : result = False
			
		return result

	def is_unknown(self,x,y,size=0.3): #in meters		
	
		if self.map is None : return False
		#reverse x,y in map image
		x_map = y - self.map_00_pose[0]
		y_map = x - self.map_00_pose[1]

		x_map = int( x_map / self.map_res )
		y_map = int( y_map / self.map_res )
		size_map = max(1, int( size/self.map_res ))
		
		#max_val = np.amax( self.map_img[x_map-size_map:x_map+size_map,y_map-size_map:y_map+size_map] )
		min_val = np.amin( self.map_img[x_map-size_map:x_map+size_map,y_map-size_map:y_map+size_map] )
		#print '[is_unknown] x,y,min_val : ' ,x,y, min_val
		if min_val == -1 : result = True
		else : result = False
			
		return result

	def approach(self,dest,dist,wait=True,clear_costmap=False,timeout = 60,dist_inc=0.05,use_sight_clearing=False,allow_unknown=True):

		if clear_costmap : self.map_clear_srv()
		
		if dest[0] == 0 and dest[1] == 0 : return False

		robot_pose,robot_angle = self.get_loc()

		delta = (robot_pose[0]-dest[0])**2+(robot_pose[1]-dest[1])**2

		goal_idx = -1
		goal_dist = 99999
		dists = []
		for i in range(6):
			dists.append( dist + i*dist_inc)

		for d in dists :
			num_sample = 30
			angles = 2*np.pi*np.arange(num_sample)/float(num_sample)
			dx = d*np.cos(angles)
			dy = d*np.sin(angles)
			goal_angles = angles+np.pi		
			spots = np.zeros((num_sample,2))
			spots[:,0] = dest[0]+dx
			spots[:,1] = dest[1]+dy
			dx2 = min(d,0.5)*np.cos(angles)
			dy2 = min(d,0.5)*np.sin(angles)


			for i in range(num_sample):
				if self.is_occupied(spots[i,0],spots[i,1],0.3,allow_unknown) : continue
				if (robot_pose[0]-spots[i,0])**2+(robot_pose[1]-spots[i,1])**2 < goal_dist :
					div_num = min(200, max(2, int(d/0.01) ) )
					xs = np.linspace(spots[i,0],dest[0]+dx2[i],div_num)
					ys = np.linspace(spots[i,1],dest[1]+dy2[i],div_num)

					ok = True
					if use_sight_clearing :
						for j in range(div_num):
							if self.is_occupied(xs[j],ys[j],0.2) :
								ok = False
								break
					if ok:
						goal_dist = (robot_pose[0]-spots[i,0])**2+(robot_pose[1]-spots[i,1])**2
						goal_idx = i
					
			if goal_idx != -1 : break
		if goal_idx == -1 : 
			print '[PIO] Approach Failed : no valid goal -> move to goal directly'
			yaw = self.quat_to_yaw(robot_angle)
			result = self.go_to_goal(dest[0],dest[1],yaw,wait,clear_costmap,timeout)
			return result
				
				
		result = self.go_to_goal(spots[goal_idx,0],spots[goal_idx,1],goal_angles[goal_idx],wait,clear_costmap,timeout)

		robot_pose,robot_angle = self.get_loc()
		delta = (robot_pose[0]-dest[0])**2+(robot_pose[1]-dest[1])**2
		
		if delta <= (dist+0.2)**2 and delta >= (dist-0.2)**2 :
			print '[PIO] Approach : aleady there'
			self.stop()
			return True

		if delta < (dist-0.2)**2 :
			print '[PIO] Approach : too close'
			self.stop()
			return True

		if result == 0 :
			return True
		else : 
			return False


	def get_loc(self,p=np.array([0,0,0]),o=np.array([0,0,0,1]),source='CameraTop_frame',target='map'):#pose = np.array([x,y,z]) : position w.r.t. robot
		pp = PoseStamped()
		pp.pose.position.x = p[0]
		pp.pose.position.y = p[1]
		pp.pose.position.z = p[2]
		pp.pose.orientation.x = o[0]
		pp.pose.orientation.y = o[1]
		pp.pose.orientation.z = o[2]
		pp.pose.orientation.w = o[3]
		#pp.header.stamp = rospy.get_rostime()
		pp.header.frame_id = source #'CameraDepth_frame'

		self.transform.waitForTransform(target,source,time=rospy.Time(),timeout=rospy.Duration(3.0))
		asdf = self.transform.getLatestCommonTime(target,source)
		pp.header.stamp = asdf

		result = self.transform.transformPose(target,pp)
		result_p = np.array([result.pose.position.x,result.pose.position.y,result.pose.position.z])
		result_o = np.array([result.pose.orientation.x,result.pose.orientation.y,result.pose.orientation.z,result.pose.orientation.w])
		return result_p, result_o

	def quat_to_yaw(self,quat):
		rpy=transformations.euler_from_quaternion((quat[0],quat[1],quat[2],quat[3]))
		yaw = rpy[2]
		return yaw

	def yaw_to_quat(self,yaw):
		quat=transformations.quaternion_from_euler(0,0,yaw)
		return quat

	def callback_objs(self,msg):
		self.objects = msg
		self.comm_delay = msg.comm_delay
		self.objs_history.append(self.objects)
		self.objs_history_idx.append(self.objects.msg_idx)
		if len(self.objs_history) > self.objs_history_max :
			del self.objs_history[  0  ] ; 	del self.objs_history_idx[  0  ]


	def callback_waving(self,msg):
		self.people_waving = msg
		self.pose_history.append(msg)
		self.pose_history_idx.append(msg.msg_idx)
		if len(self.pose_history) > self.pose_history_max :
			del self.pose_history[  0  ] ; 	del self.pose_history_idx[  0  ]

	def callback_identify(self,msg):
		self.people_identified = msg
		self.reid_history.append(msg)
		self.reid_history_idx.append(msg.msg_idx)
		if len(self.reid_history) > self.reid_history_max :
			del self.reid_history[  0  ] ; 	del self.reid_history_idx[  0  ]

	def callback_caps(self,msg):
		self.people_captioned = msg
		self.cap_history.append(msg)
		self.cap_history_idx.append(msg.msg_idx)
		if len(self.cap_history) > self.cap_history_max :
			del self.cap_history[  0  ] ; 	del self.cap_history_idx[  0  ]

	def get_objects(self,fil=None):
		self.record_image()
		objs = self.objects
		if fil is None : return objs
		else :
			result = objs_array()
			result.header = objs.header
			result.msg_idx = objs.msg_idx
			result.scene_rgb = objs.scene_rgb
			result.scene_cloud = objs.scene_cloud

			for item in objs.objects :
				if item.class_string in fil :
					result.objects.append(item)

			return result

	def get_objects_w_loc(self,fil=None) : return self.get_objects(fil)

	def get_people_wavings(self,waving_only=False,time_limit = 3.0):
		self.record_image()
		people = self.people_waving
		td =  time.time() - people.header.stamp.secs
		#print 'wav',td
		if td > time_limit :
			#print 'get_people_wavings - timeout',td
			return objs_array() #TODO good?

		if not waving_only : return people
		else:
			result = objs_array()
			result.header = people.header
			result.msg_idx = people.msg_idx
			result.scene_rgb = people.scene_rgb
			result.scene_cloud = people.scene_cloud
			for item in people.objects :
				if item.isWaving == 1 : result.objects.append(item)
			return result

	def get_people_identified(self,name=None,time_limit = 2.0):
		self.record_image()

		people = self.people_identified
		td =  time.time() - people.header.stamp.secs
		#print 'id',td
		if td > time_limit :
			#print 'get_people_identified - timeout',td ;
			return objs_array() #TODO good?
		if name is None : return people
		else:
			result = objs_array()
			result.header = people.header
			result.msg_idx = people.msg_idx
			result.scene_rgb = people.scene_rgb
			result.scene_cloud = people.scene_cloud
			for item in people.objects :
				if item.person_name == name : result.objects.append(item)
			return result


	def init_captioning(self): #deprecated
		return None


	def get_captions(self):
		self.record_image()

		msg = objs_array()
		msg.scene_rgb = self.numpyimg_to_rosimg( self.get_rgb() )

		self.captioning_req_time = time.time()
		result = ['no description']
		self.captioning_flag = False
		self.pub_cap_req.publish(msg)
		while time.time()-self.captioning_req_time < 3 :
			#print time.time()-self.captioning_req_time
			if self.captioning_flag : result = self.captioning_result ; break
		self.captioning_flag = False
		return result
	
	def get_vqa(self,question,img=None):
		self.record_image()

		msg = objs_array()
		if img is None : img = self.get_rgb()
		msg.scene_rgb = self.numpyimg_to_rosimg( img )
		if question[-1] != '?' : question+='?'
		msg.tags.append(question)

		self.vqa_req_time = time.time()
		result = 'no answer'
		self.vqa_flag = False
		self.pub_vqa_req.publish(msg)
		while time.time()-self.vqa_req_time < 3 :
			#print time.time()-self.captioning_req_time
			if self.vqa_flag : result = self.vqa_result ; break
		self.vqa_flag = False
		return result

	def publish_image(self,img,publisher):
		publisher.publish(   self.cvbridge.cv2_to_imgmsg(img,"bgr8")   )

	def callback_captioning(self,msg):
		self.captioning_result = msg.tags
		self.captioning_flag = True
	def callback_vqa(self,msg):
		if len(msg.tags) < 2 : 
			print 'invalid number of tags in VQA'
			return None
		self.vqa_result = msg.tags[-1]
		self.vqa_flag = True

	def add_external_process(self,cmd,path="./",title="ext"):
		full_cmd = 'gnome-terminal --tab --command="'+ cmd + '" --title="' + title +'"'
		print full_cmd
		self.ext_process_modules.append( pexpect.spawn(full_cmd, cwd=path,timeout = 10) )

	def callback_obstacle(self,msg):
		if msg[0] != 'Safety' : return None

		if self.reflex_going : return None

		self.reflex_going = True

		obstacle_loc = self.get_loc( msg[2] , source='odom',target='base_footprint' )[0]

		x_dir = 1 - 2 * int(obstacle_loc[0]>0)
		y_dir = 1 - 2 * int(obstacle_loc[1]>0)
		x_mag = 0.2
		y_mag = 0.1
		w_mag = 0.2

		vel_x = x_mag * x_dir
		#vel_y = y_mag * y_dir
		vel_w = w_mag * y_dir

		#print obstacle_loc
		#print vel_x, vel_y
		self.set_velocity(vel_x,0,0,0.5)
		self.set_velocity(0,0,vel_w,0.85)
		#self.set_velocity(0,0,-vel_w,1)
		time.sleep(0.5)
		self.reflex_going = False


	def global_localization(self,timelimit=60):
		self.say('start localization')
		dummy = self.localization_srv()
		time.sleep(2)
		tt = time.time()
		cov_norm = np.linalg.norm( self.pose_cov )
		self.set_velocity(0.,0.,0.5)
		while time.time()-tt < timelimit and cov_norm > self.global_localization_cov_threshold :
			time.sleep(1)
			cov_norm = np.linalg.norm( self.pose_cov )
		self.set_velocity(0.,0.,0.)
		print 'Localization : cov_norm = ' , cov_norm, ' time = ', (time.time()-tt)
		self.say('localization done')


	def callback_pose(self,msg):
		self.position = np.array( [ msg.pose.pose.position.x , msg.pose.pose.position.y , msg.pose.pose.position.z ] )
		self.orientation = np.array( [ msg.pose.pose.orientation.x , msg.pose.pose.orientation.y , msg.pose.pose.orientation.z , msg.pose.pose.orientation.w ] )
		self.pose_cov = np.array( msg.pose.covariance )

	def create_nav_goal(self,x, y, yaw):
		mb_goal = MoveBaseGoal()
		mb_goal.target_pose.header.frame_id = '/map' # Note: the frame_id must be map
		mb_goal.target_pose.pose.position.x = x
		mb_goal.target_pose.pose.position.y = y
		mb_goal.target_pose.pose.position.z = 0.0 # z must be 0.0 (no height in the map)

		# Orientation of the robot is expressed in the yaw value of euler angles
		angle = yaw # angles are expressed in radians
		quat = quaternion_from_euler(0.0, 0.0, angle) # roll, pitch, yaw
		mb_goal.target_pose.pose.orientation = Quaternion(*quat.tolist())

		return mb_goal

	def init_speech_recognition(self,sensitivity=0.3):
		# sound detection
		self.enable_speech_recog = True
		self.sound_det_s = self.session.service("ALSoundDetection")
		self.sound_det_s.setParameter("Sensitivity", sensitivity)
		self.sound_det_s.subscribe('sd')
		self.sound_det = self.ALMEM.subscriber("SoundDetected")
		self.sound_det.signal.connect(self.callback_sound_det)
		#self.thread_headfix.start()

		self.say('Speech recognition start')
		self.sr_flag = True


	def set_sound_sensitivity(self,sensitivity=0.3):
		self.sound_det_s.setParameter("Sensitivity", sensitivity)

	def callback_sound_det(self,msg):
		ox = 0		
		for i in range(len(msg)):
			if msg[i][1] == 1 : ox = 1
		if ox == 1 and self.enable_speech_recog:
			self.record_time = time.time()+self.record_delay
			if not self.thread_recording.is_alive() : self.start_recording(reset=True)
		else : return None

	def start_recording(self,reset=False,base_duration=3, withBeep=True):
		if reset :
			self.kill_recording_thread()
 			self.audio_recorder.stopMicrophonesRecording()
			self.record_time = time.time()+base_duration

		if not self.thread_recording.is_alive():
			self.thread_recording = Thread(target=self.record_audio,args=(self.speech_hints,withBeep))
			self.thread_recording.daemon = False
			self.thread_recording.start()
			self.thread_recording.join()
			print self.speech_memory
			return self.speech_memory
		return ''

	def kill_recording_thread(self):
		if self.thread_recording.is_alive() :
			self.audio_terminate = True
			time.sleep(0.3)
			self.audio_terminate = False

	def record_audio(self,hints, withBeep=True):
		if withBeep:
			self.audio_player.playSine(1000,self.beep_volume,1,0.3)
			time.sleep(0.5)
		print 'Speech Detected : Start Recording'
		channels = [0,0,1,0] #left,right,front,rear
		fileidx = "recog"
		self.audio_recorder.startMicrophonesRecording("/home/nao/record/"+fileidx+".wav", "wav", 48000, channels)
		while time.time() < self.record_time :
			if self.audio_terminate :
				self.audio_recorder.stopMicrophonesRecording()
				print 'kill!!'
				return None
			time.sleep(0.1)
		self.audio_recorder.stopMicrophonesRecording()
		self.audio_recorder.recording_ended = True

		if not os.path.exists('./audio_record'):
			os.mkdir('./audio_record', 0755)

		cmd = 'sshpass -p 1847! scp nao@'+str(self.ip)+':/home/nao/record/'+fileidx+'.wav ./audio_record'
		os.system(cmd)

		print 'End recording'
		self.speech_memory = self.stt2("audio_record/recog.wav",hints)
		
		if self.enable_concept_map and self.speech_memory != '' : 
			stamp = str(time.time()).replace('.','')
			spp = self.speech_memory.split(' ')			
			self.text_concepts.append([int(stamp),self.speech_memory])
			if len(self.text_concepts) > 1000 : del self.text_concepts[0]
			for ww in spp : 
				self.text_corpus.append(ww)
				
				
			self.text_corpus = list(set(self.text_corpus))
				
			f = open('concept_map/raw_data/'+stamp+'.txt','w')
			for iii in range( len(spp)-1 ) :
				f.write(spp[iii]+'\n')
			f.write(spp[-1])
			f.flush()
			f.close
			
			self.thread_concept = Thread(target=self.record_concepts,args=(stamp,spp,))
			self.thread_concept.daemon = False
			self.thread_concept.start()

				
		if self.data_recording and self.data_recording_dir != '' :
			now = datetime.datetime.now()
			strnow = now.strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
			shutil.copy2('audio_record/recog.wav' , self.data_recording_dir + '/audio/AUPAIR_AUDIO_'+ strnow + '.wav')
		if withBeep:
			self.audio_player.playSine(250,self.beep_volume,1,0.3)
			time.sleep(1)
		return None



	def get_rgb(self): #return current RGB image from camera (top camera)
		msg = self.video.getImageRemote(self.rgb_top)
		w = msg[0]
		h = msg[1]
		c = msg[2]
		data = msg[6]
		ba = str(bytearray(data))


		nparr = np.fromstring(ba, np.uint8)
		img_np = nparr.reshape((h,w,c))
		img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

		return img_np

	def get_depth(self): #return current Depth image from camera (Xtion)
		msg = self.video.getImageRemote(self.depth)

		w = msg[0]
		h = msg[1]
		c = msg[2]
		data = msg[6]
		ba = str(bytearray(data))

		nparr = np.fromstring(ba, np.uint16)
		img_np = nparr.reshape((h,w,1))
		#img_np = cv2.cvtself.head_fixColor(img_np, cv2.COLOR_RGB2BGR)

		return img_np

	def get_pos_wrt_robot(self,x,y,size=10,scan_len=50,scan='point'):
		#scan : point(around), vertical(line)
		self.point_clouds = self.get_point_cloud(0)
		if scan == 'point':
			x1 = min(240, max(0, x - size//2) )
			x2 = min(240, max(0, x + size//2) )
			y1 = min(320, max(0, y - size//2) )
			y2 = min(320, max(0, y + size//2) )

			roi = self.point_clouds[x1:x2,y1:y2]
			mask = roi[:,:,0]>0
			masked = roi[mask]
			if masked.size == 0 : return np.array([0,0,0])
			mask = masked[:,0]==masked[:,0].min()
			masked = masked[mask]
			return masked[0]#self.point_clouds[x,y]
		else :
			xx1 = min(240,max(0,x-scan_len))
			xx2 = min(240,max(0,x+scan_len))

			roi = self.point_clouds[xx1:xx2,y-2:y+2,:]
			mask = roi[:,:,0]>0
			masked = roi[mask]
			if masked.size == 0 : return np.array([0,0,0])
			mask = masked[:,0]==masked[:,0].min()
			masked = masked[mask]
			return masked[0]#self.point_clouds[x,y]

	def get_point_cloud(self,dummy): #return point cloud from depth image
		msg = self.video.getImageRemote(self.depth)

		w = msg[0]
		h = msg[1]
		c = msg[2]
		data = msg[6]
		ba = str(bytearray(data))

		nparr = np.fromstring(ba, np.uint16)
		img_np = nparr.reshape((h,w,1))
		offx = 0 ; offy = 0
		img_np[:img_np.shape[0]-offx,:] = img_np[offx:,:]
		img_np[:,:img_np.shape[1]-offy] = img_np[:,offy:]

		fx = 525.0 * 0.54 ; fy = 525.0 * 0.54 ;
		cx = 319.5; cy = 239.5;

		z = img_np/1000.0

		x = (self.x_temp) * z / fx
		y = (self.y_temp) * z / fy

		img_np = np.zeros((240,320,3))

		xx = 200
		yy = 300

		img_np[:,:,0] = z[:,:].reshape(240,320)
		img_np[:,:,1] = y[:,:].reshape(240,320)
		img_np[:,:,2] = x[:,:].reshape(240,320)
		return img_np

	def do_anim(self,command,block=True): #string command
		#play motion animation (bowing, waving)
		#avalable animation list : http://doc.aldebaran.com/2-4/naoqi/motion/alanimationplayer-advanced.html#alanimationplayer-advanced
#		self.animation.run(command)
		result = self.animation.run('animations/Stand/'+command,_async=True)
		if block :	result.value()

	def do_pose(self,command): #string command
		#Set to fixed pose
		#avalable pose list : Crouch,LyingBack,LyingBelly,Sit,SitRelax,Stand,StandInit,StandZero.
		self.posture.goToPosture(command,1.0)
		#self.say(command)
		#print self.posture.getPostureFamily()

	def callback_sound_loc(self,msg):
		#process sound localization
		self.sound_loc_data = msg

	def get_sound_loc(self): #last direction of detected sound
		return self.sound_loc_data[1][0] #[[sec,milisec],[azimuth(rad), elevation(rad), confidence, energy],[etc]]

	def go_to_goal_odom(self,p_wrt_odom,o_wrt_odom,clear_costmap=False,costmap_clear_interval=10.,timeout=60): #z=yaw in radian
		#  p and o are numpy arrays
		#print 'Go to (Odom) '
		if clear_costmap : self.map_clear_srv()
		tic = time.time()
		last_clear = time.time()
		while time.time()-tic < timeout :
			goal_p,goal_o = self.get_loc(p_wrt_odom,o_wrt_odom,source='odom',target='map')
			goal_yaw = self.quat_to_yaw(goal_o)
			if clear_costmap and time.time()-last_clear > costmap_clear_interval : self.map_clear_srv() ; last_clear = time.time()
			result = self.go_to_goal(goal_p[0],goal_p[1],goal_yaw,wait=True,clear_costmap=False,timeout=1.0)
			if result == 0 : return 0
		self.stop()
		return 1

	def go_to_goal(self,x,y,theta,wait=True,clear_costmap=False,timeout=0): #z=yaw in radian
		if x==0 and y==0 : return 1
		#print 'Go to ' + str(x) + ',' +str(y)
		if clear_costmap : self.map_clear_srv()
		nav_goal = self.create_nav_goal(x ,y, theta)
		self.nav_as.send_goal(nav_goal)

		if wait :
			if timeout > 0 :
				self.nav_as.wait_for_result(timeout=rospy.Duration(timeout))
			else :
				self.nav_as.wait_for_result()

			nav_res = self.nav_as.get_result()
			nav_state = self.nav_as.get_state()
			if nav_state == 3 : print "moved to " + str(x) + ',' + str(y) + ',' + str(theta) ; return 0
			elif nav_state ==4 : print "failed to move to " + str(x) + ',' + str(y) + ',' + str(theta) ; return 1
			elif nav_state == 5 : print "goal is not atainable!" ; return 1
			else : return 1
		else : return -1

	def set_velocity(self,x,y,theta,duration=-1.): #m/sec, rad/sec
		#if duration > 0 : stop after duration(sec)
		tt = Twist()
		tt.linear.x = x
		tt.linear.y = y
		tt.angular.z = theta
		self.cmd_vel_pub.publish(tt)
		if duration < 0 : return None
		tic = time.time()
		while time.time() - tic < duration :
			self.cmd_vel_pub.publish(tt)
			time.sleep(0.1)
		tt = Twist()
		tt.linear.x = 0
		tt.linear.y = 0
		tt.angular.z = 0
		self.cmd_vel_pub.publish(tt)

	def cancel_plan(self) : self.goal_cancel_pub.publish( GoalID() )

	def stop(self) : self.cancel_plan() ; self.set_velocity(0,0,0)

	def activate_keyboard_control(self):
		print ("### keyboard control ###")
		print ("usage : type following command and press enter.")
		print ("		robot will maintain velocity unless you give another command.")
		print ("commands:")
		print ("	 w : forward")
		print ("	 s : stop")
		print ("	 x : backward")
		print ("	 a : strafe left")
		print ("	 d : strafe right")
		print ("	 q : turn left")
		print ("	 e : turn right")
		print ("	 ss : say next input")
		print ("	 lw : load waypoints from file")
		print ("	 aw : add current location to waypoints")
		print ("	 gw : go to waypoint")
		print ("	 sw : save all waypoints")
		print ("	 ar : add reid target")
		print ("	 sr : save reid target")
		print ("	 cr : clear reid targets")
		print ("	 re : speech recognition")
		print ("	 c : deactivate keyboard control")
		print ("	 so : save all objects in sight")
		print ("	 cm : clear costmap")
		print ("	 exit : kill!")
		print ("########################")
		self.say("start keyboard control")
		command = ''
		while command != 'c' :
			command = raw_input('next command : ')
			if command == 'w' : self.set_velocity(0.25,0,0)
			elif command == 's' : self.stop()
			elif command == 'x' : self.set_velocity(-0.25,0,0)
			elif command == 'a' : self.set_velocity(0,0.25,0)
			elif command == 'd' : self.set_velocity(0,-0.25,0)
			elif command == 'q' : self.set_velocity(0,0,0.35)
			elif command == 'e' : self.set_velocity(0,0,-0.35)
			elif command == 'qq' : self.set_velocity(0,0,1)
			elif command == 'ee' : self.set_velocity(0,0,-1)
			elif command == 'exit' : self.callback_kill()
			elif command == 'ss' :
				say_string = raw_input('Say : ')
				self.say(say_string)
			elif command == 'aw' :
				way_string = raw_input('waypoint name : ')
				self.add_waypoint(way_string)
			elif command == 'lw' :
				way_string = raw_input('waypoint name : ')
				self.load_waypoints(way_string)
			elif command == 'gw' :
				way_string = raw_input('waypoint name : ')
				self.go_to_waypoint(way_string,wait=True,clear_costmap=True)
			elif command == 'sw' :
				fn_string = raw_input('file name : ')
				self.save_waypoints(fn_string)

			elif command == 'ar' :
				visible_objs = self.get_objects(['person'])
				if len(visible_objs.objects) > 1 :
					self.say('only one person should be in sight')
					self.say('there are '+str(len(visible_objs.objects)) + ' people')
					continue
				elif len(visible_objs.objects) < 1 :
					self.say('cannot find person') ; continue
				idx_string = raw_input('person index : ')
				name_string = raw_input('person name : ')

				self.add_reid_target(int(idx_string),name_string,img=None,person_obj=visible_objs.objects[0])
				self.say('added '+name_string)
			elif command == 'sr' :
				self.save_reid_targets()
			elif command == 'cr' :
				self.clear_reid_targets()
			elif command == 're' :
				self.start_recording(reset=True)
			elif command == 'lre' :
				duration = input('duration: ')
				self.start_recording(True, duration)
				
			elif command == 'cm' : 
				self.map_clear_srv()

			elif command == 'so' :
				self.say('saving all objects')
				per = self.get_perception()
				idxx = 0
				for item in per.objects :
					img = self.rosimg_to_numpyimg(item.cropped)
					fn = item.class_string + '_' + str(time.time())
					cv2.imwrite('./object_images/'+fn+str(idxx)+'.png',img)
					idxx+=1
			elif command == 'c' : self.set_velocity(0,0,0) ; break
			else : print("Invalid Command!")
		self.say("stop keyboard control")

	def go_in_meter(self,x,y,clear_costmap=False):
		#Makes the robot navigate to a relative position
		loc = get_loc(np.array([x,y,0]))
		self.go_to_goal(loc[0],loc[1],0,clear_costmap)
		return None

	def say(self,text): #TODO kill current recording!
		#say
		if self.muted: return
		print '[PIO] say : ' + text
		self.kill_recording_thread()
		if self.sr_flag : self.enable_speech_recog = False
		time.sleep(0.15)
		self.tts.say(text)
		time.sleep(0.15)
		if self.sr_flag : self.enable_speech_recog = True

	def set_volume(self,volume): #(0.0~1.0)
		self.tts.setVolume(volume)

	def stt2(self,speech_file, hints=[]):
		tic = time.time()
		"""Transcribe the given audio file asynchronously."""
		with io.open(speech_file, 'rb') as audio_file:
		    content = audio_file.read()
		    sample = self.speech_client.sample(
			content,
			source_uri=None,
			encoding='LINEAR16',
			sample_rate_hertz=48000)
		try:
			if len(hints) > 0 :
				alternatives = sample.recognize(language_code='en-US', max_alternatives=10, speech_contexts=hints,)
			else :
				alternatives = sample.recognize(language_code='en-US', max_alternatives=1, speech_contexts=hints,)

			result = alternatives[0].transcript.lower()
			max_score = 0

			for alt in alternatives :
				score = 0
				for hint in hints :
					if self.find_word(hint.lower() , alt.transcript.lower()) :
						score += 1
				if score > max_score :
					result = alt.transcript.lower()
					max_score = score


			if self.data_recording and self.data_recording_dir != '' :
				self.record_command(result)
			print 'srec : ' ,time.time()-tic
			return result

		except :
			print 'fail to recognize'
			return ''


	def rosimg_to_numpyimg(self,img_msg,encoding='bgr8'):
		return self.cvbridge.imgmsg_to_cv2(img_msg,encoding)

	def numpyimg_to_rosimg(self,npimg,encoding='bgr8'):
		return self.cvbridge.cv2_to_imgmsg(npimg,encoding)

	def find_word(self,word,source=None):
		if source is None : source = self.speech_memory
		sm_seg = source.split()
		for w in sm_seg :
			if w == word : return True
		return False

	def follow_person(self,target,target_dist=1.0,fail_dist_threshold=1.0,dist_strict=False, \
					  timeout=60,stop_criterion='dist',use_reid=False,reid_name=None,stop_word='stop',\
					  reid_strict=False,reid_add=False,score_threshold=-10,max_fail_count = 20, short_mode_thr = 1.5):
		#stop criterion : dist, speech(stop)
		self.stop()
		result = False #false : fail, true : success
		fail_count = 0 #fail if cannot detect target for 10 times
		dist_threshold = fail_dist_threshold #fail if target moves more than this meter in following frame
		robot_trj = []

		tx = target.pose_wrt_map.position.x
		ty = target.pose_wrt_map.position.y

		reid_base_idx = 99
		reid_max_count = 50
		reid_count = 0
		reid_add_interval = 1 #sec
		reid_last_add = time.time()
		if reid_name is None : reid_name = 'following'

		for i in range(reid_base_idx, reid_base_idx + reid_max_count):
			self.remove_reid_target(i)

		if use_reid:
			self.add_reid_target(reid_base_idx+reid_count,reid_name,person_obj=target)
			target.reid_score = 1.0
			reid_count += 1
			reid_last_add = time.time()

		tic = time.time()
		target_trajectory = []
		#cv2.imwrite( 'before.png', self.rosimg_to_numpyimg(target.cropped) )
		waypoint_timer = time.time()
		while time.time()-tic < timeout :
			if fail_count >  max_fail_count : self.stop() ; return False , target_trajectory ,robot_trj
			if result and stop_criterion=='dist' : self.stop() ; return True , target_trajectory, robot_trj
			if stop_criterion=='speech' and self.find_word(stop_word) : self.stop() ; return True , target_trajectory, robot_trj
			if time.time() - waypoint_timer < 0.2:
				trj_odom_p,trj_odom_o = self.get_loc(target='odom')
				robot_trj.append([trj_odom_p,trj_odom_o])
				waypoint_timer = time.time()


			tx = target.pose_wrt_map.position.x
			ty = target.pose_wrt_map.position.y

			# get next target
			next_target,total_score, scores = self.get_nearest_target(target,dist_threshold,(0.1,0.1,0.05,1.0))
			if next_target is not None :
				xx1,yy1 = next_target.pose_wrt_odom.position.x , next_target.pose_wrt_odom.position.y
				xx2,yy2 = target.pose_wrt_odom.position.x , target.pose_wrt_odom.position.y
				print xx1, yy1, xx2, yy2
				if dist_strict and (next_target.valid_pose != 1 or target.valid_pose != 1 or (xx1-xx2)**2+(yy1-yy2)**2 > dist_threshold**2):
					next_target = None
		

			if (reid_strict and scores[3] != 1.0) or next_target is None or total_score < score_threshold :
				print 'target lost'
				time.sleep(0.2)
				if fail_count == 0 : self.set_velocity(0,0,0)
				fail_count += 1
				continue

			target = next_target
			#cv2.imwrite( 'after.png', self.rosimg_to_numpyimg(target.cropped) )

			tx_r = target.pose_wrt_robot.position.x
			ty_r = target.pose_wrt_robot.position.y

			# 4. navigate using only bounding box if map position is not available
			if tx == 0 and ty == 0 :
				self.stop()
				tx_r = 10
				w = 0 ; v=0
				if target.y > 190 : w = -0.1
				elif target.y < 130 : w = 0.1
				if tx_r > target_dist+0.2 : v = 0.25
				elif tx_r < target_dist-0.2 : v = -0.25
				self.map_clear_srv
				self.set_velocity(v,0,w)
				fail_count += 1
				time.sleep(1.0)
				continue

			# 5. reset fail_count if succeed to find target
			fail_count = 0

			delta = tx_r**2+ty_r**2
			if delta < short_mode_thr**2 :
				#prFint 'close mode', target.y, tx_r
				self.cancel_plan()
				v = 0 ; w = 0 ; y_reached = 1 ; x_reached = 1
				# y=320 : 30, y=191 : 10
				if target.y > 190 : w = -np.pi/18. - np.pi*(target.y-190.)/(140*18) ; y_reached = 0
				elif target.y < 130 : w = np.pi/18. + np.pi*(130. - target.y)/(130*18) ; y_reached = 0
				if tx_r > target_dist+0.2 : v = 0.28 ; x_reached = 0
				elif tx_r < target_dist-0.2 : v = -0.28 ; x_reached = 0
				self.set_velocity(v,0,w)
				result =  x_reached * y_reached
				time.sleep(0.1)
			else :
				#print 'long mode'
				result = self.approach(  (tx,ty)  , target_dist , False , True  )
				target_trajectory.append(target)
				time.sleep(0.2)

		return False , target_trajectory, robot_trj

	def get_nearest_target(self,target,dist_limit=0.5,weights=(1.0,0.1,0.05,1.0)):

		obs = self.get_perception([target.class_string])
		closest = None
		max_score = -9999999
		hist_score = -1 ; loc_score = -1 ; box_size_score = -1 ; reid_score = -1

		for o in obs.objects :
			hist_score =  self.compare_hist( self.rosimg_to_numpyimg(target.cropped) , self.rosimg_to_numpyimg(o.cropped) )
			loc_score = -10
			if target.valid_pose == 1 and o.valid_pose == 1 :
				temp = (target.pose_wrt_map.position.x-o.pose_wrt_map.position.x)**2 + (target.pose_wrt_map.position.y-o.pose_wrt_map.position.y)**2
				if temp < dist_limit**2:
					loc_score = -temp
			box_size_score = - min(100, (target.w*target.h-o.w*o.h)**2)
			reid_score = 0
			if target.person_name != '' and target.person_name == o.person_name : reid_score = 1.0

			score = weights[0]*hist_score + weights[1]*loc_score + weights[2]*box_size_score + weights[3]*reid_score
			if score > max_score :
				max_score = score
				closest = o

		return closest, max_score, (hist_score,loc_score,box_size_score,reid_score)

	def compare_hist (self,img1,img2): #img1, img2 : numpy array of cropped image
		img2 = cv2.resize(img2,(img1.shape[1] , img1.shape[0]))
		img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
		img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)

		hist0 = cv2.calcHist([img1],[0, 1], None, [10, 16], [0,180,0,256])
		hist1 = cv2.calcHist([img2],[0, 1], None, [10, 16], [0,180,0,256])
		score = cv2.compareHist(hist0,hist1,0) #method 0~6
		#print score
		return score

	def find_target(self,fil=[],talk='',reid_name=None,timeout=30,speed=0.2,direction_change=False,direction_change_interval=5,dist_limit = 3,waypoint_ignore=None,wp_ignore_dist=2.3,waypoint_only=None,only_dist=3,allow_unknown = False):
		#type_ = list that contains 'waving' or 'sitting' or 'reid' or 'standing'.  if empty, find any person.
		tictic = time.time()
		if len(fil)==0 : fil = ['person']
		self.say(talk)
		tic =time.time()
		direction = 1
		last_change = time.time()
		dci = float(direction_change_interval) / 2
		while time.time()-tic < timeout:
			per = self.get_perception()
			if len( per.objects ) > 0 : self.set_velocity(0,0,speed*direction)
			else : self.set_velocity(0,0,speed*direction*2)

			for item in per.objects :
				ok_flag = False
				for a in fil :
					if a in item.tags : ok_flag = True ; break
				
				if ok_flag and item.valid_pose == 1 and item.pose_wrt_robot.position.x < dist_limit :
					
					xxx = item.pose_wrt_map.position.x
					yyy = item.pose_wrt_map.position.y	
					
					unk = False
					if not allow_unknown :
						unk = self.is_unknown(xxx,yyy)
					if unk : continue
					
					if waypoint_ignore is not None :
						wpig_flag = True
						for wwpp in waypoint_ignore :
							xxw = self.waypoints[wwpp][0]
							yyw = self.waypoints[wwpp][1]
							print wwpp, xxx, yyy, xxw, yyw, (xxx-xxw)**2 + (yyy-yyw)**2 
							if (xxx-xxw)**2 + (yyy-yyw)**2 < wp_ignore_dist**2 : 
								wpig_flag = False
								break
						if not wpig_flag : continue
							
					if waypoint_only is not None :					
						wpon_flag = False
						for wwpp in waypoint_only :
							xxw = self.waypoints[wwpp][0]
							yyw = self.waypoints[wwpp][1]
							print wwpp, xxx, yyy, xxw, yyw, (xxx-xxw)**2 + (yyy-yyw)**2 
							if (xxx-xxw)**2 + (yyy-yyw)**2 < only_dist**2 : 
								wpon_flag = True
								break
						if not wpon_flag : continue
							
					self.stop()
					self.set_velocity(0,0,-speed*direction,0.3)
					time.sleep(1)

					target,total_score,scores = self.get_nearest_target(item)

					print total_score,scores
					if target is not None :
						print 'Found target'
						self.say('Found target')
						target.isWaving = item.isWaving
						target.person_id = item.person_id
						target.person_name = item.person_name
						target.isSitting = item.isSitting

						print time.time()-tictic
						return target

			if direction_change and time.time()-last_change > dci :
				direction *= -1 ; dci = direction_change_interval ; last_change = time.time()
		self.stop()
		print 'failed ', time.time()-tictic
		return None

	def rotate_in_radian(self,angle=np.pi):
		self.stop()
		#self.set_velocity(0,0,1,1)
		half = 8.0
		speed = 0.4
		if angle > np.pi:
			angle = -(2 * np.pi - angle)
		if angle < 0 : speed = -0.4

		rotate_time = half * abs(angle) / np.pi
		self.set_velocity(0, 0, speed, rotate_time)
		
		
	def record_concepts(self,timestamp,spp,timelimit=5,interval=1):
		iii = 0
		jjj = 0
		tic = time.time()
		while time.time() - tic < timelimit :
			objs = self.get_perception().objects
			annotated = self.annotated_scene.copy()
			self.img_concepts.append([int(timestamp),annotated])
			cv2.imwrite('concept_map/raw_data/'+timestamp+'_'+str(jjj)+'.png',annotated)
			jjj+=1
			if len(self.img_concepts) > 1000 : del self.img_concepts[0]
			for o in objs :
				self.obj_concepts.append([int(timestamp),o])
				cv2.imwrite('concept_map/raw_data/'+timestamp+'_'+o.class_string+'_'+str(iii)+'.png',self.rosimg_to_numpyimg(o.cropped))
				iii+=1
				if len(self.obj_concepts) > 1000 : del self.obj_concepts[0]

			time.sleep(interval)
		#self.update_concept_map(spp,)		
		
	def update_concept_map(self):
		return None

	def wait_until_touch(self,part='Head',timelimit=300):
		flag = False
		self.touch_stat = []
		tic = time.time()
		while not flag and time.time()-tic < timelimit:
			ts = self.touch_stat
			if len(ts) > 0 :
				if ts[0][0] == part and ts[0][1] :
					print '[PIO] ' + part + ' touch detected'
					flag = True
			#speak something
			if flag : break
		return None
		

def main():

	params = pepper_config.load_config()

	pio = pepper_io(params)
	pio.set_volume(1.0)
	pio.speech_hints = params['person_names']
	pio.activate_keyboard_control()
	

if __name__ == "__main__":
	main()
