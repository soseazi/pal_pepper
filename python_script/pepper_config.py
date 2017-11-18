import argparse

def load_config():
	
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", required = False, default='192.168.125.11', help = "ip")
	ap.add_argument("-p", "--port", required = False, default='9559', help = "port")
	args = vars(ap.parse_args())	
	
	nao_ip = args['ip']
	nao_port = args['port']

	config = {
		#NaoQI
		'ip' : nao_ip ,
		'port' : nao_port,
		#PIO
		'show_integrated_perception' : True,
		#Obj Detector
		'rgb_topic' : 'pepper_robot/camera/front/image_raw',
		'obj_topic' : 'objects',
		'od_show' : False,		
		#Reid
		'reid_target_topic' : 'reid_targets',
		'reid_topic' : 'people_identified',
		'reid_thr' : 0.75,
		#Pose
		'pose_topic' : 'people_w_pose',
		'waving_thr' : 10, #pixel
		'sitting_thr' : -0.05, #meter
		#Captioning
		'captioning_topic' : 'objects_w_caption',
		
		#Common in all scenarios
		'person_names' : ['emma','noah','olivia','liam','sophia','mason','ava','jacob','isabella','william','mia','ethan','abigail','james','emily','alexander','charlotte','michael','harper','benjamin'],		
		
	}
	return config
