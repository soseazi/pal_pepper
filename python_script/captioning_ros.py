import os
import re
import sys
import cv2
import math
import time
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
from tf import TransformListener
from std_srvs.srv import Empty
from pepper_jychoi.msg import objs, objs_array
#pexpect
import pexpect
 
 
class captioner:
	
	def __init__(self,obj_topic=None,caption_topic=None):
		self.OBJ_TOPIC = 'objects_loc'	
		self.CAPTION_TOPIC = 'objects_caption'	
		if obj_topic  is not None : self.OBJ_TOPIC = obj_topic
		if caption_topic  is not None : self.CAPTION_TOPIC = caption_topic
		self.sub_obj = rospy.Subscriber(self.OBJ_TOPIC, objs_array, self.callback_objs, queue_size=1)
		self.caption_pub = rospy.Publisher(self.CAPTION_TOPIC,objs_array,queue_size=1)
		self.last_detect = time.time()
		self.ext_process_modules = []
		return None

	def add_external_process(self,cmd,path="./",title="ext"):
		full_cmd = 'gnome-terminal --tab --command="'+ cmd + '" --title="' + title +'"'
		print full_cmd
		self.ext_process_modules.append( pexpect.spawn(full_cmd, cwd=path,timeout = 10) )

	def init_captioning(self):
		#Captioning process
		cmd_launch_captioning = "th run_ros.lua"	
		path_captioning = "./captioning"	
		self.add_external_process(cmd_launch_captioning,path=path_captioning,title='captioning')
		self.pub_cap_req = rospy.Publisher('captioning_request',Image,queue_size=1)
		self.sub_caption = rospy.Subscriber("captioning_result", String, self.callback_captioning)
		self.captioning_req_time = time.time()
		self.captioning_flag = False
		self.captioning_result = []

	

	def callback_act(self,msg):
		if msg.data == 1 : self.flag = 1

	def callback_objs(self,msg):
		#if self.flag == 0 : return None
		if time.time() - self.last_detect < 2 : return None
		tic = time.time()
		humans = []
		human_imgs = []
		pubmsg = objs_array()
		temp = []
		for i in range(len(msg.objects)):
			if msg.objects[i].class_string == 'person':
				humans.append(i)
				img = self.cvbridge.imgmsg_to_cv2(msg.objects[i].cropped,'bgr8')
				imgg = np.zeros((200,200,3))
				if img.shape[0] >= img.shape[1]:
					scale = 200.0/img.shape[0]
				else :
					scale = 200.0/img.shape[1]
				img2 = cv2.resize(img, (0,0), fx = scale, fy=scale, interpolation=cv2.INTER_CUBIC)
				imgg[ :img2.shape[0] , :img2.shape[1] ] = img2.copy()
				human_imgs.append(   img   )
		minibatch = human_imgs
		#print minibatch.shape
		#print len(human_imgs)
		for ii in range(len(human_imgs)):
			hansup = self.detect(human_imgs[ii])
			msg.objects[i].isWaving = hansup
			temp.append(msg.objects[i])
			print ii, hansup
		pubmsg.header=msg.header
		pubmsg.objects = temp
		self.pose_pub.publish(pubmsg)
		toc = time.time()
		print toc-tic
		self.last_detect = time.time()
		self.flag = 0

		

	def detect(self,oriImg):	

		tic = time.time()
		imageToTest = Variable(T.transpose(T.transpose(T.unsqueeze(torch.from_numpy(oriImg).float(),0),2,3),1,2),volatile=True).cuda()
	 
		multiplier = [x * self.model_['boxsize'] / oriImg.shape[0] for x in self.param_['scale_search']]
		 
		heatmap_avg = torch.zeros((len(multiplier),19,oriImg.shape[0], oriImg.shape[1])).cuda()
		paf_avg = torch.zeros((len(multiplier),38,oriImg.shape[0], oriImg.shape[1])).cuda()
		#print heatmap_avg.size()
		tic2 = time.time()
		for m in range(len(multiplier)):
		    scale = multiplier[m]
		    h = int(oriImg.shape[0]*scale)
		    w = int(oriImg.shape[1]*scale)
		    pad_h = 0 if (h%self.model_['stride']==0) else self.model_['stride'] - (h % self.model_['stride']) 
		    pad_w = 0 if (w%self.model_['stride']==0) else self.model_['stride'] - (w % self.model_['stride'])
		    new_h = h+pad_h
		    new_w = w+pad_w
		 
		    imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
		    imageToTest_padded, pad = util.padRightDownCorner(imageToTest, self.model_['stride'], self.model_['padValue'])
		    imageToTest_padded = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5
		    
		    feed = Variable(T.from_numpy(imageToTest_padded)).cuda()      
		    output1,output2 = self.model(feed)
		    heatmap = nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])).cuda()(output2)
		    
		    paf = nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])).cuda()(output1)       
		 
		    heatmap_avg[m] = heatmap[0].data
		    paf_avg[m] = paf[0].data  

		heatmap_avg = T.transpose(T.transpose(T.squeeze(T.mean(heatmap_avg, 0)),0,1),1,2).cuda() 
		paf_avg     = T.transpose(T.transpose(T.squeeze(T.mean(paf_avg, 0)),0,1),1,2).cuda() 
		heatmap_avg=heatmap_avg.cpu().numpy()
		paf_avg    = paf_avg.cpu().numpy()
		 
		all_peaks = []
		peak_counter = 0
		 
		#maps = 
		for part in range(18):
		    map_ori = heatmap_avg[:,:,part]
		    map = gaussian_filter(map_ori, sigma=3)
		    
		    map_left = np.zeros(map.shape)
		    map_left[1:,:] = map[:-1,:]
		    map_right = np.zeros(map.shape)
		    map_right[:-1,:] = map[1:,:]
		    map_up = np.zeros(map.shape)
		    map_up[:,1:] = map[:,:-1]
		    map_down = np.zeros(map.shape)
		    map_down[:,:-1] = map[:,1:]
		    
		    peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > self.param_['thre1']))
		#    peaks_binary = T.eq(
		#    peaks = zip(T.nonzero(peaks_binary)[0],T.nonzero(peaks_binary)[0])
		    
		    peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
		    
		    peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
		    id = range(peak_counter, peak_counter + len(peaks))
		    peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
		 
		    all_peaks.append(peaks_with_score_and_id)
		    peak_counter += len(peaks)

		result = 0

		if len(all_peaks[1]) > 0:
			if len(all_peaks[4]) > 0:
				if all_peaks[4][0][1] < all_peaks[1][0][1] : result = 1
			if len(all_peaks[7]) > 0:
				if all_peaks[7][0][1] < all_peaks[1][0][1] : result = 1

		    


	        print('Neck : ', all_peaks[1])
	        print('R_hand : ', all_peaks[4])
	        print('L_hand : ', all_peaks[7])
		
		canvas = oriImg # B,G,R order
		for i in [1,4,7]:  # for i in range(18):
		    for j in range(len(all_peaks[i])):
			#print i,j,all_peaks[i][j]
		        cv2.circle(canvas, all_peaks[i][j][0:2], 4, self.colors[i], thickness=-1)
		cv2.imwrite('pose.png',canvas)
		cv2.waitKey(1)
		return result
		

def main():
	rospy.init_node("pose_detector")	
	pd = pose_detector()

        rospy.spin()


if __name__=='__main__':	
	main()
