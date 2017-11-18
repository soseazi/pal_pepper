import os
import re
import sys
import cv2
import math
import time
import scipy
import argparse
import matplotlib
from torch import np
import pylab as plt
from joblib import Parallel, delayed
import util_pose as util
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from pose_config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
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
from tf import TransformListener, Transformer
from std_srvs.srv import Empty
from pal_pepper.msg import objs, objs_array
import pepper_config
import tensorflow
import qi
from threading import Thread

'''
0 nose
1 neck
2 r_shoulder
3 r_elbow
4 r_wrist
5 l_shoulder
6 l_elbow
7 l_wrist
8 r_pelvis
9 r_knee
10 r_anckle
11 l_pervis
12 l_knee
13 l_ankle
14 r_eye
15 l_eye
16 r_ear
17 l_ear
'''
 
torch.set_num_threads(torch.get_num_threads())
weight_name = './pose_model/pose_model.pth'
 
blocks = {}
 
# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
	   [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
	   [1,16], [16,18], [3,17], [6,18]]
	   
# the middle joints heatmap correpondence
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
	  [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
	  [55,56], [37,38], [45,46]]
	  
# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
	  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
	  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
	 
block0  = [{'conv1_1':[3,64,3,1,1]},{'conv1_2':[64,64,3,1,1]},{'pool1_stage1':[2,2,0]},{'conv2_1':[64,128,3,1,1]},{'conv2_2':[128,128,3,1,1]},{'pool2_stage1':[2,2,0]},{'conv3_1':[128,256,3,1,1]},{'conv3_2':[256,256,3,1,1]},{'conv3_3':[256,256,3,1,1]},{'conv3_4':[256,256,3,1,1]},{'pool3_stage1':[2,2,0]},{'conv4_1':[256,512,3,1,1]},{'conv4_2':[512,512,3,1,1]},{'conv4_3_CPM':[512,256,3,1,1]},{'conv4_4_CPM':[256,128,3,1,1]}]
 
blocks['block1_1']  = [{'conv5_1_CPM_L1':[128,128,3,1,1]},{'conv5_2_CPM_L1':[128,128,3,1,1]},{'conv5_3_CPM_L1':[128,128,3,1,1]},{'conv5_4_CPM_L1':[128,512,1,1,0]},{'conv5_5_CPM_L1':[512,38,1,1,0]}]
 
blocks['block1_2']  = [{'conv5_1_CPM_L2':[128,128,3,1,1]},{'conv5_2_CPM_L2':[128,128,3,1,1]},{'conv5_3_CPM_L2':[128,128,3,1,1]},{'conv5_4_CPM_L2':[128,512,1,1,0]},{'conv5_5_CPM_L2':[512,19,1,1,0]}]
 
for i in range(2,7):
	blocks['block%d_1'%i]  = [{'Mconv1_stage%d_L1'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L1'%i:[128,128,7,1,3]},
{'Mconv5_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L1'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L1'%i:[128,38,1,1,0]}]
	blocks['block%d_2'%i]  = [{'Mconv1_stage%d_L2'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L2'%i:[128,128,7,1,3]},
{'Mconv5_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L2'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L2'%i:[128,19,1,1,0]}]
 
def make_layers(cfg_dict):
	layers = []
	for i in range(len(cfg_dict)-1):
		one_ = cfg_dict[i]
		for k,v in one_.iteritems():
			if 'pool' in k:
				layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
			else:
				conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
				layers += [conv2d, nn.ReLU(inplace=True)]
	one_ = cfg_dict[-1].keys()
	k = one_[0]
	v = cfg_dict[-1][k]
	conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
	layers += [conv2d]
	return nn.Sequential(*layers)

layers = []
for i in range(len(block0)):
	one_ = block0[i]
	for k,v in one_.iteritems():  
		if 'pool' in k:
			layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
		else:
			conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
			layers += [conv2d, nn.ReLU(inplace=True)]  
       
models = {}	   
models['block0']=nn.Sequential(*layers)	
 
for k,v in blocks.iteritems():
	models[k] = make_layers(v)
		
class pose_model(nn.Module):
	def __init__(self,model_dict,transform_input=False):
		super(pose_model, self).__init__()
		self.model0   = model_dict['block0']
		self.model1_1 = model_dict['block1_1']	
		self.model2_1 = model_dict['block2_1']  
		self.model3_1 = model_dict['block3_1']  
		self.model4_1 = model_dict['block4_1']  
		self.model5_1 = model_dict['block5_1']  
		self.model6_1 = model_dict['block6_1']  
	
		self.model1_2 = model_dict['block1_2']	
		self.model2_2 = model_dict['block2_2']  
		self.model3_2 = model_dict['block3_2']  
		self.model4_2 = model_dict['block4_2']  
		self.model5_2 = model_dict['block5_2']  
		self.model6_2 = model_dict['block6_2']

	
	def forward(self, x):    
		out1 = self.model0(x)
	
		out1_1 = self.model1_1(out1)
		out1_2 = self.model1_2(out1)
		out2  = torch.cat([out1_1,out1_2,out1],1)
	
		out2_1 = self.model2_1(out2)
		out2_2 = self.model2_2(out2)
		out3   = torch.cat([out2_1,out2_2,out1],1)
	
		out3_1 = self.model3_1(out3)
		out3_2 = self.model3_2(out3)
		out4   = torch.cat([out3_1,out3_2,out1],1)
	 
		out4_1 = self.model4_1(out4)
		out4_2 = self.model4_2(out4)
		out5   = torch.cat([out4_1,out4_2,out1],1)  
	
		out5_1 = self.model5_1(out5)
		out5_2 = self.model5_2(out5)
		out6   = torch.cat([out5_1,out5_2,out1],1)	 
		      
		out6_1 = self.model6_1(out6)
		out6_2 = self.model6_2(out6)
	
		return out6_1,out6_2	
 
class pose_detector:
	# visualize
	colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
	  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
	  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
	model = pose_model(models)     
	model.load_state_dict(torch.load(weight_name))
	model.cuda()
	model.float()
	model.eval()	 
	param_, model_ = config_reader()
	def __init__(self,params):
		self.transform = TransformListener()
		self.transformer = Transformer(True,rospy.Duration(10.0))
		self.params = params

		self.cvbridge = CvBridge()
		self.OBJ_TOPIC = self.params['obj_topic']
		self.POSE_TOPIC = self.params['pose_topic']	

		self.sub_obj = rospy.Subscriber(self.OBJ_TOPIC, objs_array, self.callback_objs, queue_size=1)
		self.pose_pub = rospy.Publisher(self.POSE_TOPIC,objs_array,queue_size=1)
		self.flag = 0
		self.last_detect = time.time()
		

		return None

	def callback_act(self,msg):
		if msg.data == 1 : self.flag = 1

	def callback_objs(self,msg):
		#if self.flag == 0 : return None
		
		tic = time.time()
		humans = []
		human_imgs = []
		scales = []
		pubmsg = msg
		
		for item in msg.objects :
			if item.class_string == 'person' :
				item.joints = [-1]*36
				humans.append(item)
				img = self.cvbridge.imgmsg_to_cv2(item.cropped,'bgr8')
				imgg = np.zeros((self.model_['boxsize'],self.model_['boxsize'],3))
				if img.shape[0] >= img.shape[1]:
					scale = float(self.model_['boxsize'])/img.shape[0]
				else :
					scale = float(self.model_['boxsize'])/img.shape[1]
				scales.append(scale)
				img2 = cv2.resize(img, (0,0), fx = scale, fy=scale, interpolation=cv2.INTER_CUBIC)
				imgg[ :img2.shape[0] , :img2.shape[1] ] = img2.copy()
				human_imgs.append(   imgg   )		
	
		if len(humans) == 0 :
			self.pose_pub.publish(msg)
			return None
		minibatch = np.stack( human_imgs , axis=0)
		print 'minibatch_shape : ' , minibatch.shape
		num_run = minibatch.shape[0] // 20
		humans_updated = []
		for i in range(num_run+1) : 
			s = i*20
			e = min(minibatch.shape[0],(i+1)*20)
			output = self.detect(minibatch[s:e],humans[s:e],scales[s:e])
			humans_updated += output
		print 'elapsed time : ' ,time.time()-tic
		tic = time.time()

		humans_updated=self.post_processing(humans_updated)
			
		pubmsg.objects = humans_updated
		self.pose_pub.publish(pubmsg)
		toc = time.time()
		print 'elapsed time : ' ,toc-tic
		print ''
		self.last_detect = time.time()
		self.flag = 0

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
		#print rospy.Time() 
		self.transform.waitForTransform(target,source,time=rospy.Time(),timeout=rospy.Duration(3.0))
		asdf = self.transform.getLatestCommonTime(target,source)
		pp.header.stamp = asdf

		result = self.transform.transformPose(target,pp)
		result_p = np.array([result.pose.position.x,result.pose.position.y,result.pose.position.z])
		result_o = np.array([result.pose.orientation.x,result.pose.orientation.y,result.pose.orientation.z,result.pose.orientation.w])
		return result_p, result_o
	
	def post_processing(self, humans):
		for h in humans :
			'''
			0 1 nose
			2 3 neck
			4 5 r_shoulder
			6 7 r_elbow
			8 9 r_wrist
			10 11 l_shoulder
			12 13 l_elbow
			14 15 l_wrist
			16 17 r_pelvis
			18 19 r_knee
			20 21 r_anckle
			22 23 l_pervis
			24 25 l_knee
			26 27 l_ankle
			28 29 r_eye
			30 31 l_eye
			32 33 r_ear
			34 35 l_ear
			'''
			#rasing hand
			if h.joints[10] >= 0 and h.joints[12] >= 0 and h.joints[10] > h.joints[12]:
				h.isLWaving = 1
				h.tags.append('lwaving')

			if h.joints[4] >= 0 and h.joints[6] >= 0 and h.joints[4] > h.joints[6]:
				h.isRWaving = 1
				h.tags.append('rwaving')
			
			if h.isLWaving == 1 or h.isRWaving == 1:
				h.isWaving = 1
				h.tags.append('waving')
			'''
			#rasing hand
			if h.joints[10] >= 0 and h.joints[14] >= 0 and h.joints[10] > h.joints[14]+self.params['waving_thr']:
				h.isLWaving = 1
				h.tags.append('lwaving')

			if h.joints[4] >= 0 and h.joints[8] >= 0 and h.joints[4] > h.joints[8]+self.params['waving_thr']:
				h.isRWaving = 1
				h.tags.append('rwaving')
			
			if h.isLWaving == 1 or h.isRWaving == 1:
				h.isWaving = 1
				h.tags.append('waving')
			'''
			#sitting
			shoulder_h = -9999
			cropped_cloud = self.cvbridge.imgmsg_to_cv2(h.cropped_cloud,desired_encoding="passthrough")
			'''
			#adjust positions
			valid_joints = []
			for i in range(0,18):
				if h.joints[2*i] != -1 and h.joints[2*i+1] != -1 :
					temp = self.get_pos_wrt_robot(cropped_cloud , h.joints[2*i] , h.joints[2*i+1])
					if (temp != 0).all():
						valid_joints.append( temp )
			if len(valid_joints) > 0 and h.valid_pose == 1 :
				pos_wrt_robot = np.array(valid_joints)
				pos_wrt_robot = np.median(pos_wrt_robot,axis=0)
				print 'pose_wrt_robot : ' , pos_wrt_robot
				h.pose_wrt_robot.position.x = pos_wrt_robot[0]
				h.pose_wrt_robot.position.y = pos_wrt_robot[1]
				h.pose_wrt_robot.position.z = pos_wrt_robot[2]
				pose_wrt_map = self.get_loc(pos_wrt_robot)
				print pose_wrt_map
				h.pose_wrt_map.position.x = pose_wrt_map[0]
				h.pose_wrt_map.position.y = pose_wrt_map[1]
				h.pose_wrt_map.position.z = pose_wrt_map[2]
				pose_wrt_odom = self.get_loc(pos_wrt_robot,target='odom')
				h.pose_wrt_odom.position.x = pose_wrt_odom[0]
				h.pose_wrt_odom.position.y = pose_wrt_odom[1]
				h.pose_wrt_odom.position.z = pose_wrt_odom[2]				
			'''

			
			if h.joints[4] > 0 and h.joints[5] > 0 : 
				shoulder_h = max(shoulder_h, self.get_pos_wrt_robot(cropped_cloud , h.joints[4] , h.joints[5])[2])
			if h.joints[10] > 0 and h.joints[11] > 0 : 				
				shoulder_h = max(shoulder_h, self.get_pos_wrt_robot(cropped_cloud , h.joints[10] , h.joints[11])[2])
			if shoulder_h < self.params['sitting_thr'] and shoulder_h > -9999 : 
				h.isSitting = 1
				h.tags.append('sitting')

			# lying and standing
			knee = np.mean([h.joints[18], h.joints[24]])
			if h.joints[2] >= 0 and knee >= 0 and abs(h.joints[2] - knee) <= 20:
				h.isLying = 1
				h.tags.append('lying')

			# pointing
			point_length = 45
			if h.joints[9] >= 0 and h.joints[5] >= 0 and abs(h.joints[5] - h.joints[9]) > point_length:
				if h.joints[5] - h.joints[9] > 0:
					h.isRPointing = 1 ; h.tags.append('rpointing')
				else:
					h.isLPointing = 1 ; h.tags.append('lpointing')

			if h.joints[11] >= 0 and h.joints[15] >= 0 and abs(h.joints[15] - h.joints[11]) > point_length:
				if h.joints[15] - h.joints[11] > 0:
					h.isLPointing = 1 ; h.tags.append('lpointing')
				else:
					h.isRPointing = 1 ; h.tags.append('rpointing')
		
		
			print h.object_id, h.person_name, h.isWaving, h.isSitting, shoulder_h	
		return humans
		
	def get_pos_wrt_robot(self,cropped_cloud,x,y,size=10,scan_len=50,scan='point'):
		#scan : point(around), vertical(line)
		h = cropped_cloud.shape[0]
		w = cropped_cloud.shape[1]
		if scan == 'point':
			x1 = min(h, max(0, x - size//2) )
			x2 = min(h, max(0, x + size//2) )
			y1 = min(w, max(0, y - size//2) )
			y2 = min(w, max(0, y + size//2) )

			roi = cropped_cloud[x1:x2,y1:y2]
			mask = roi[:,:,0]>0
			masked = roi[mask]
			if masked.size == 0 : return np.array([0,0,0])
			mask = masked[:,0]==masked[:,0].min()
			masked = masked[mask]
			return masked[0]#self.point_clouds[x,y]
		else :
			xx1 = min(h,max(0,x-scan_len))
			xx2 = min(h,max(0,x+scan_len))

			roi = cropped_cloud[xx1:xx2,y-2:y+2,:]
			mask = roi[:,:,0]>0
			masked = roi[mask]
			if masked.size == 0 : return np.array([0,0,0])
			mask = masked[:,0]==masked[:,0].min()
			masked = masked[mask]
			return masked[0]#self.point_clouds[x,y]		

	def detect(self,minibatch,humans,scales):	
		ms = minibatch.shape
		tic = time.time()
		imageToTest = Variable(  T.transpose(  T.transpose(  (torch.from_numpy(minibatch).float()/256.0)-0.5  ,  2,3)  ,  1,2  )  ,volatile=True).cuda()
  
		output1 , output2 = self.model(imageToTest)
	
		heatmap_avg = nn.UpsamplingBilinear2d((ms[1],ms[2])).cuda()(output2)

		paf_avg = nn.UpsamplingBilinear2d((ms[1],ms[2])).cuda()(output1)   	
	
		heatmap_avg = T.transpose( heatmap_avg , 1,2)
		heatmap_avg = T.transpose( heatmap_avg , 2,3)

		heatmap_avg=heatmap_avg.cpu().data.numpy()
		print 'heatmap shape : ' , heatmap_avg.shape

		paf_avg = paf_avg.cpu().data.numpy()
	
		heatmap_avg = heatmap_avg[:,:,:,:-1]
	
		map =  gaussian_filter1d(heatmap_avg, sigma=3 , axis=2)
		map = gaussian_filter1d(map, sigma=3 , axis=1)
	
		map_left = np.zeros(map.shape)
		map_left[:,1:,:,:] = map[:,:-1,:,:]
		map_right = np.zeros(map.shape)
		map_right[:,:-1,:,:] = map[:,1:,:,:]
		map_up = np.zeros(map.shape)
		map_up[:,:,1:,:] = map[:,:,:-1,:]
		map_down = np.zeros(map.shape)
		map_down[:,:,:-1,:] = map[:,:,1:,:]
		
		peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > self.param_['thre1']))
		peaks = np.nonzero(peaks_binary)
		
		for i in range(peaks[0].shape[0]):
			human_idx = int(peaks[0][i])
			
			x = int(  peaks[1][i] / scales[human_idx]  )
			y = int(  peaks[2][i] / scales[human_idx]  )
			#x = peaks[1][i]
			#y = peaks[2][i]
			joint = peaks[3][i]
			humans[human_idx].joints[joint*2] = x
			humans[human_idx].joints[joint*2+1] = y
	
		return humans	
		#tt = 0
		

		

def main():
	rospy.init_node("pose_detector")	
	params = pepper_config.load_config()
	pd = pose_detector(params)
	print 'Pose Detector Ready!'
	rospy.spin()


if __name__=='__main__':	
	main()
