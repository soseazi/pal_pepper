from darkflow.net.build import TFNet
import numpy as np
import cv2
import time
import sys
import argparse
import os

#ROS modules
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion , Twist, Pose, PoseStamped, Vector3
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int32,String
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from tf import TransformListener, Transformer, transformations
from std_srvs.srv import Empty
from pal_pepper.msg import objs, objs_array
import pepper_config
import tensorflow
import qi
from threading import Thread


class obj_detector:
	options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.3 ,"gpu":0.2 , "summary":None}
	
	class_reroute = {
		"remote" : "bottle",
		"cup" : "bottle",
		"pizza" : "cake",
		"teddybear":"cake",
	}
		
	def __init__(self,params):
		#NaoQI
		self.ip = params['ip'] ; self.port = params['port']
		self.session = qi.Session()	
		
		self.match_images = {}
		
		ld = os.listdir('./object_images/')
		
		for fn in ld :
			if fn.split('.')[-1] != 'png' or len(fn.split('_'))!=3 : continue
			on = fn.split('.')[0]
			img = cv2.imread('./object_images/'+fn)
			self.match_images[on] = img
		
		print 'matching objects loaded : ' , self.match_images.keys()
		
		try:
			self.session.connect("tcp://" + self.ip + ":" + str(self.port))
		except RuntimeError:
			print ("Connection Error!")
			sys.exit(1)
		self.video = self.session.service("ALVideoDevice")
		video_subs_list = ['detector_rgb_t_0','detector_pc_0','detector_dep_0']
		print self.video.getSubscribers()
		for sub_name in video_subs_list :
			print sub_name, self.video.unsubscribe(sub_name)

		self.rgb_top = self.video.subscribeCamera('detector_rgb_t',0,2,11,20)   #name, idx, resolution, colorspace, fps
		self.pc = self.video.subscribeCamera('detector_pc',2,1,19,20)
		self.depth = self.video.subscribeCamera('detector_dep',2,1,17,20)
		
		self.point_cloud = np.zeros((240,320,3))
		offset_x = -120
		offset_y = -160
		self.x_temp = -(np.tile(np.arange(240).reshape(240,1) , (1,320)).reshape(240,320,1) + offset_x)
		self.y_temp = -(np.tile(np.arange(320).reshape(1,320) , (240,1)).reshape(240,320,1) + offset_y)
		
		#self.thread_cloud = Thread(target=self.get_point_cloud, args=(None,))
		#self.thread_cloud.daemon = True
		#self.thread_cloud.start()	
		time.sleep(1)
		print self.video.getSubscribers()		
		
		#Darknet
		self.gg = tensorflow.Graph()		
		with self.gg.as_default() as g:		
			self.tfnet = TFNet(self.options)
		self.classes = open('cfg/coco.names','r').readlines()
		
		#ROS
		self.cvbridge = CvBridge()		
		
		self.transform = TransformListener()
		self.transformer = Transformer(True,rospy.Duration(10.0))

		self.RGB_TOPIC = params['rgb_topic']
		self.OBJ_TOPIC = params['obj_topic']

		self.rgb_sub = rospy.Subscriber(self.RGB_TOPIC, Image, self.callback_image, queue_size=1)
		self.obj_pub = rospy.Publisher(self.OBJ_TOPIC,objs_array,queue_size=1)

		self.show = params['od_show']
		self.object_id = 0
		self.object_id_max = 999999
		self.msg_idx = 0
		self.msg_idx_max = 9999999
		if self.show : 
			cv2.startWindowThread()
			cv2.namedWindow('objs')
		self.tttt =time.time()
		time.sleep(1)

		
		#self.thread_cloud = Thread(target=self.callback_image, args=(None,))
		#self.thread_cloud.daemon = True
		#self.thread_cloud.start()
		
	
	def compare_hist (self,img1,img2): 
		img1 = cv2.resize(img1,(32,32))
		img2 = cv2.resize(img2,(32,32))
		img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
		img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)

		hist0 = cv2.calcHist([img1],[0, 1], None, [10, 16], [0,180,0,256])
		hist1 = cv2.calcHist([img2],[0, 1], None, [10, 16], [0,180,0,256])
		score = cv2.compareHist(hist0,hist1,0) #method 0~6
		#print score
		return score

		
	def get_point_cloud(self,dummy): #return point cloud from depth image
			tic = time.time()
			msg = self.video.getImageRemote(self.depth)
			#print time.time()-tic
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
			
	def callback_image(self,msg):

		tic = time.time()
		img = self.cvbridge.imgmsg_to_cv2(msg, 'bgr8')
		self.point_clouds = self.get_point_cloud(0)
		#print 'get_point : ' , time.time()-tic
		img = cv2.resize(img, (320,240) )
		detections = self.tfnet.return_predict(img)
		img_display = img.copy()

		objarray = objs_array()
		objarray.comm_delay = time.time()-self.tttt
		print 'detection : ' ,time.time()-self.tttt
		self.tttt = time.time()
		objarray.header = msg.header
		objarray.header.stamp = rospy.Time.from_sec(time.time())
		objarray.msg_idx = self.msg_idx
		self.msg_idx += 1
		if self.msg_idx > self.msg_idx_max : self.msg_idx = 0		
		temp = []
		objarray.header.frame_id = 'CameraTop_frame'
		
		temp_tt = 0
		
		for i in range(len(detections)):
			obj = objs()
			obj.object_id = self.object_id
			self.object_id += 1
			if self.object_id > self.object_id_max : self.object_id = 0
			obj.person_id = -1 #unknown
			obj.person_name = ''
			obj.class_string = detections[i]['label']
			obj.tags.append(detections[i]['label'])
			if obj.class_string == 'person' : obj.tags.append('people')
			tlx = int(detections[i]['topleft']['y'])
			tly = int(detections[i]['topleft']['x'])
			brx = int(detections[i]['bottomright']['y'])
			bry = int(detections[i]['bottomright']['x'])
			
			x = (tlx + brx)/2
			y = (tly + bry)/2
			h = (brx - tlx)/2
			w = (bry - tly)/2

			obj.x = x
			obj.y = y
			obj.h = h
			obj.w = w
			obj.confidence = detections[i]['confidence']
			
			crop = img[ max(0,x-h) : min(img.shape[0],x+h) , max(0,y-w) : min(img.shape[1],y+w) ]
			
			ttiicc = time.time()
			max_score = -1
			sub_class = None
			for mi in self.match_images.keys() :
				mi_spl = mi.split('_')
				mi_cls = mi_spl[0]
				mi_subcls = mi_spl[1]
				mi_idx = mi_spl[2]
				ob_cls = obj.class_string
				if mi_cls in self.class_reroute.keys():
					mi_cls = self.class_reroute[mi_cls]
				if ob_cls in self.class_reroute.keys():
					ob_cls = self.class_reroute[ob_cls]
				if ob_cls != mi_cls : continue
				scr = self.compare_hist(crop,self.match_images[mi])
				#print mi, scr,
				if max_score < scr : 
					max_score = scr
					sub_class = mi_subcls
			#print ''
			temp_tt += time.time()-ttiicc
			if sub_class is not None : obj.tags.append(sub_class)
				
			if self.show:
				cv2.rectangle(img_display,(tly,tlx),(bry,brx),(0,255,0),2)
				lbl = detections[i]['label'] if sub_class is None else sub_class
				cv2.putText(img_display,lbl,(tly,tlx-15),cv2.FONT_HERSHEY_SIMPLEX,0.3,color=(0,0,0),thickness=1)
			
			
			obj.cropped = self.cvbridge.cv2_to_imgmsg(crop,"bgr8")			
			cropped_point = self.point_clouds[obj.x-obj.h : obj.x+obj.h , obj.y-obj.w : obj.y+obj.w ]
			obj.cropped_cloud = self.cvbridge.cv2_to_imgmsg(cropped_point,encoding="passthrough") 

			point_x = min( max(0, int(obj.x - 0.5*obj.h) ) , 240 )
			
			pose_wrt_robot = self.get_pos_wrt_robot(point_x,obj.y,scan_len=obj.h,scan='point')
			if (pose_wrt_robot == 0).all() : continue
			if pose_wrt_robot[0] > 8.0 : continue #max range = 10m??
			obj.pose_wrt_robot.position.x = pose_wrt_robot[0]
			obj.pose_wrt_robot.position.y = pose_wrt_robot[1]
			obj.pose_wrt_robot.position.z = pose_wrt_robot[2]
			pose_wrt_map = self.get_loc(pose_wrt_robot)[0]
			obj.pose_wrt_map.position.x = pose_wrt_map[0]
			obj.pose_wrt_map.position.y = pose_wrt_map[1]
			obj.pose_wrt_map.position.z = pose_wrt_map[2]
			pose_wrt_odom = self.get_loc(pose_wrt_robot,target='odom')[0]
			obj.pose_wrt_odom.position.x = pose_wrt_odom[0]
			obj.pose_wrt_odom.position.y = pose_wrt_odom[1]
			obj.pose_wrt_odom.position.z = pose_wrt_odom[2]
			obj.valid_pose = 1
			
			temp.append(  obj   )
		
		#print temp_tt
		objarray.objects = temp
		objarray.scene_rgb = msg
		objarray.scene_cloud = self.cvbridge.cv2_to_imgmsg(self.point_clouds,'passthrough')

		if self.show:
			cv2.imshow('objs',cv2.resize(img_display,(640,480)))
		self.obj_pub.publish(objarray)
		#print 'detection_process : ' , time.time()-tic
	

	def get_pos_wrt_robot(self,x,y,size=10,scan_len=50,scan='point'):
		#scan : point(around), vertical(line)
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
	
def main():
	params = pepper_config.load_config()	
	rospy.init_node("object_detector")	
	yolo = obj_detector(params)
	rospy.spin()


if __name__=='__main__':	
	main()
