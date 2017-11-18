# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Implementation-CVPR2015-CNN-for-ReID
# Copyright (c) 2017 Ning Ding
# Licensed under The MIT License [see LICENSE for details]
# Written by Ning Ding
# --------------------------------------------------------

"""
Model Definition and Compile Script.
"""
from keras.layers import Input
from keras.layers.core import Lambda,Flatten,Dense
from keras.layers.convolutional import Conv2D,UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add,Concatenate
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import cv2

import time
import argparse
import numpy as np
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
from tf import TransformListener
from std_srvs.srv import Empty
from pal_pepper.msg import objs, objs_array
import pepper_config



class person_identifier:

	def __init__(self,params):
		self.params = params
		self.threshold = self.params['reid_thr']
		self.model = self.generate_model()
		self.model = self.compile_model(self.model)
		weight='weights/reid_weight.h5'
		with self.gg.as_default() as g:
			self.model.load_weights(weight)
		self.OBJ_TOPIC = self.params['obj_topic']
		self.TARGET_TOPIC = self.params['reid_target_topic']
		self.PERSON_ID_TOPIC = self.params['reid_topic']

		self.cvbridge = CvBridge()
		self.sub_target = rospy.Subscriber(self.TARGET_TOPIC, objs_array, self.callback_targets, queue_size=1)
		self.sub_obj = rospy.Subscriber(self.OBJ_TOPIC, objs_array, self.callback_objs, queue_size=1)
		self.reid_pub = rospy.Publisher(self.PERSON_ID_TOPIC,objs_array,queue_size=1)

		self.targets = objs_array()
		self.last_id = 0
		print('Start identification module')


	def identify(self,minibatch): #0:diff person, 1:same person
		with self.gg.as_default() as g:
			result = self.model.predict_on_batch(minibatch)
			result = np.array(result)
			return result[:,1]

	def process_message(self,obj_list):
		if len(obj_list) == 0 : 
			return np.zeros((1,160,60,3))
		result = np.zeros(  (len(obj_list),160,60,3)   )
		for i in range(len(obj_list)):
			result[i,:,:,:] = self.proc_img(  self.cvbridge.imgmsg_to_cv2(obj_list[i].cropped,'bgr8')   )
		return result		

	def callback_targets(self,msg):
		print '#### Targets ####'		
		self.targets = msg
		self.targets_names = {} ; 	self.targets_names2 = {}
		next_name_idx = 0
		for ob in self.targets.objects :
			if ob.class_string == 'person' : 
				if not ob.person_name in self.targets_names.keys() : 
					self.targets_names[ob.person_name] = next_name_idx 
					self.targets_names2[next_name_idx] = ob.person_name
					next_name_idx += 1
		print self.targets_names.keys()
		self.targets_np = self.process_message(self.targets.objects)

	def callback_objs(self,msg):
		
		pubmsg = msg

		if len(self.targets.objects) > 0 : 
			targets = self.targets
			targets_np = self.targets_np.copy()

			tic = time.time()
			humans = []

			for i in range(len(msg.objects)):
				if msg.objects[i].class_string == 'person':
					humans.append(msg.objects[i])	

			if len(humans) > 0 : 
				human_imgs = self.process_message(humans)

				minibatch, indexes = self.make_minibatch(targets_np,human_imgs)

				#print indexes
				minibatch_length = minibatch[0].shape[0]

				num_run = (minibatch_length // 64)+1

				for j in range(num_run):
					end_idx = min(minibatch_length,(j+1)*64)
					small_batch = [minibatch[0][j*64:end_idx] , minibatch[1][j*64:end_idx]]
					if j == 0 :	scores = self.identify(small_batch)    
					else : 
						tt = self.identify(small_batch)
						scores = np.concatenate(   ( scores, tt) ,axis=0 )

				scores_reshape = scores.reshape((human_imgs.shape[0],targets_np.shape[0]))
				
				scores_filtered = np.zeros((human_imgs.shape[0],len(self.targets_names)))
				for i in range(scores_reshape.shape[0]) :
					for j in range(scores_reshape.shape[1]) :
						temp_idx = self.targets_names[ targets.objects[j].person_name ]
						scores_filtered[i,temp_idx] = max(scores_filtered[i,temp_idx ] , scores_reshape[i,j])
			
				for i in range(scores_filtered.shape[0]):
					sorted_idxes = np.argsort(scores_filtered[i,:])
					for j in sorted_idxes :
						if i == np.argmax(scores_filtered[:,j]) and scores_filtered[i,j] > self.threshold :
							humans[i].person_id = 0 #targets.objects[   j   ].person_id
							humans[i].person_name = self.targets_names2[j] #targets.objects[   j   ].person_name
							humans[i].reid_score = scores_filtered[i,j]		
							humans[i].tags.append( humans[i].person_name )
							break
					print humans[i].person_id,humans[i].person_name,humans[i].reid_score				
				print 'minibatch size & ellapsed time : ' , minibatch[0].shape[0] , ' ' , time.time()-tic
				print ''
				pubmsg.objects = humans
				
		self.reid_pub.publish(pubmsg)	

		return None

	def make_minibatch(self,t,h):
		num_target = t.shape[0]
		num_human = h.shape[0]
		idx_target = np.arange(num_target)
		idx_human = np.arange(num_human)

		target_tile = np.tile(t,(num_human,1,1,1))
		human_repeat = np.repeat(h,num_target,axis=0)

		idx_target_tile = np.tile(idx_target,(num_human))
		idx_human_repeat = np.repeat(idx_human,num_target,axis=0)

		idxs = np.zeros((num_target*num_human,2))
		idxs[:,0] = idx_target_tile
		idxs[:,1] = idx_human_repeat
		return [target_tile,human_repeat],idxs


	def proc_img(self,img):
		w = img.shape[1]
		scale = 60./w
		result = np.zeros((160,60,3))
		image_proc = cv2.resize(img,None,fx=scale,fy=scale)
		if image_proc.shape[0] >= 160 : result[:,:,:] = image_proc[:160,:,:]
		else : result[:image_proc.shape[0],:,:] = image_proc[:,:,:]
		result = cv2.cvtColor(result.astype('uint8'),cv2.COLOR_BGR2RGB)
		#print image_proc.shape
		result = result/255.0 #TODO check
		#image_proc = image_proc.reshape((1,160,60,3))
		#result.append(image_proc)
		return result

	def generate_model(self,weight_decay=0.0005):
	    '''
	    define the model structure
	    ---------------------------------------------------------------------------
	    INPUT:
		weight_decay: all the weights in the layer would be decayed by this factor
		
	    OUTPUT:
		model: the model structure after being defined
		
		# References
		- [An Improved Deep Learning Architecture for Person Re-Identification]
	    ---------------------------------------------------------------------------
	    '''      
	    self.gg = tf.Graph()
	    with self.gg.as_default() as g:
		    config = tf.ConfigProto()
		    #config.gpu_options.per_process_gpu_memory_fraction = 0.1
		    config.gpu_options.allow_growth = True
		    set_session(tf.Session(config=config))
		  
		    def upsample_neighbor_function(input_x):
			input_x_pad = K.spatial_2d_padding(input_x, padding=((2,2),(2,2)))
			x_length = K.int_shape(input_x)[1]
			y_length = K.int_shape(input_x)[2]
			output_x_list = []
			output_y_list = []
			for i_x in range(2, x_length + 2):
			    for i_y in range(2, y_length + 2):
				output_y_list.append(input_x_pad[:,i_x-2:i_x+3,i_y-2:i_y+3,:])
			    output_x_list.append(K.concatenate(output_y_list, axis=2))
			    output_y_list = []
			return K.concatenate(output_x_list, axis=1)
		    
		    max_pooling = MaxPooling2D()
		    
		    x1_input = Input(shape=(160,60,3))
		    x2_input = Input(shape=(160,60,3))
		    
		    share_conv_1 = Conv2D(20, 5, kernel_regularizer=l2(weight_decay), activation="relu")
		    x1 = share_conv_1(x1_input)
		    x2 = share_conv_1(x2_input)
		    x1 = max_pooling(x1)
		    x2 = max_pooling(x2)
		    
		    share_conv_2 = Conv2D(25, 5, kernel_regularizer=l2(weight_decay), activation="relu")
		    x1 = share_conv_2(x1)
		    x2 = share_conv_2(x2)
		    x1 = max_pooling(x1)
		    x2 = max_pooling(x2)
		    
		    upsample_same = UpSampling2D(size=(5, 5))
		    x1_up = upsample_same(x1)
		    x2_up = upsample_same(x2)    
		    upsample_neighbor = Lambda(upsample_neighbor_function)        
		    x1_nn = upsample_neighbor(x1)
		    x2_nn = upsample_neighbor(x2)
		    negative = Lambda(lambda x: -x)
		    x1_nn = negative(x1_nn)
		    x2_nn = negative(x2_nn)
		    x1 = Add()([x1_up, x2_nn])
		    x2 = Add()([x2_up, x1_nn])

		    conv_3_1 = Conv2D(25, 5, strides=(5, 5), kernel_regularizer=l2(weight_decay), activation="relu")
		    conv_3_2 = Conv2D(25, 5, strides=(5, 5), kernel_regularizer=l2(weight_decay), activation="relu")
		    x1 = conv_3_1(x1)
		    x2 = conv_3_2(x2)
		    
		    conv_4_1 = Conv2D(25, 3, kernel_regularizer=l2(weight_decay), activation="relu")
		    conv_4_2 = Conv2D(25, 3, kernel_regularizer=l2(weight_decay), activation="relu")
		    x1 = conv_4_1(x1)
		    x2 = conv_4_2(x2)
		    x1 = max_pooling(x1)
		    x2 = max_pooling(x2)
		    
		    y = Concatenate()([x1, x2])
		    y = Flatten()(y)   
		    y = Dense(500, kernel_regularizer=l2(weight_decay), activation='relu')(y)
		    y = Dense(2, kernel_regularizer=l2(weight_decay), activation='softmax')(y)
		    
		    model = Model(inputs=[x1_input, x2_input], outputs=[y])
		    #model.summary()
		    
		    return model

	def compile_model(self,model):
	    '''
	    compile the model after defined
	    ---------------------------------------------------------------------------
	    INPUT:
		model: model before compiled
		all the other inputs should be organized as the form 
		        loss='categorical_crossentropy'
		# Example
		        model = compiler_def(model_def,
		                             sgd='SGD_new(lr=0.01, momentum=0.9)',
		                             loss='categorical_crossentropy',
		                             metrics='accuracy')
		# Default
		        if your don't give other arguments other than model, the default
		        config is the example showed above (SGD_new is the identical 
		        optimizer to the one in reference paper)
	    OUTPUT:
		model: model after compiled
		
		# References
		- [An Improved Deep Learning Architecture for Person Re-Identification]
	    ---------------------------------------------------------------------------
	    '''    
	    with self.gg.as_default() as g:
		    class SGD_new(SGD):
			'''
			redefinition of the original SGD
			'''
			def __init__(self, lr=0.01, momentum=0., decay=0.,
				     nesterov=False):
			    super(SGD, self).__init__()
			    self.__dict__.update(locals())
			    self.iterations = K.variable(0.)
			    self.lr = K.variable(lr)
			    self.momentum = K.variable(momentum)
			    self.decay = K.variable(decay)
			    self.inital_decay = decay
		    
			def get_updates(self, params, constraints, loss):
			    grads = self.get_gradients(loss, params)
			    self.updates = []
		    
			    lr = self.lr
			    if self.inital_decay > 0:
				lr *= (1. / (1. + self.decay * self.iterations)) ** 0.75
				self.updates .append(K.update_add(self.iterations, 1))
		    
			    # momentum
			    shapes = [K.get_variable_shape(p) for p in params]
			    moments = [K.zeros(shape) for shape in shapes]
			    self.weights = [self.iterations] + moments
			    for p, g, m in zip(params, grads, moments):
				v = self.momentum * m - lr * g  # velocity
				self.updates.append(K.update(m, v))
		    
				if self.nesterov:
				    new_p = p + self.momentum * v - lr * g
				else:
				    new_p = p + v
		    
				# apply constraints
				if p in constraints:
				    c = constraints[p]
				    new_p = c(new_p)
		    
				self.updates.append(K.update(p, new_p))
			    return self.updates 
		    all_classes = {
			'sgd_new': 'SGD_new(lr=0.01, momentum=0.9)',        
			'sgd': 'SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)',
			'rmsprop': 'RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)',
			'adagrad': 'Adagrad(lr=0.01, epsilon=1e-06)',
			'adadelta': 'Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)',
			'adam': 'Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)',
			'adamax': 'Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)',
			'nadam': 'Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)',
		    }
		    param = {'optimizer': 'sgd_new', 'loss': 'categorical_crossentropy', 'metrics': 'accuracy'}
		    config = ''

		    if not len(config):
			config = all_classes[param['optimizer']]
		    optimiz = eval(config)
		    model.compile(optimizer=optimiz,
			      loss=param['loss'],
			      metrics=[param['metrics']])
		    
		    print("Model Compile Successful.")
		    return model



if __name__ == "__main__":
	rospy.init_node("person_identification")	
	params = pepper_config.load_config()
	rr = person_identifier(params)
	rospy.spin()
