# pal_pepper
This is the code for the paper

**Perception-Action-Learning System for Mobile Social-Service Robots using Deep Learning
This code has been used in Robocup@Home2017 and won 1st Place in Social Standard Platform (SSPL) AUPAIR - https://www.robocup2017.org/file/awards/Awards_RoboCup_athome.pdf
Youtube Link: https://goo.gl/Pxnf1n

If you find this code useful in your research, please cite:

```
@inproceedings{PALs,
  title={Perception-Action-Learning System for Mobile Social-Service Robots using Deep Learning},
  author={Beom-Jin Lee, Jinyoung Choi, Chung-Yeon Lee, Kyung-Wha Park, Sungjun Choi, Cheolho Han,
Dong-Sig Han, Christina Baek, Patrick Emaase, Byoung-Tak Zhang},
  booktitle={AAAI},
  year={2018}
}
```

#### Ver 1.0 (2017.11.18) by Beom-Jin Lee


### Use tmux!
install tmux
```
sudo apt-get update
sudo apt-get install -y python-software-properties software-properties-common
sudo add-apt-repository -y ppa:pi-rho/dev
sudo apt-get update
sudo apt-get install -y tmux=2.0-1~ppa1~t
```

run : `./run_pepper.sh 192.168.1.~` and run your task with IP (`task.py --ip 192.168.1.~`)

exit :`tmux kill-session`

## Additional Files
`densecap-pretrained-vgg16.t7` is excluded from git source control. It makes git too slow. You have to download the file manually at `ROBOCUP_HOME/python_script/captioning/data/models/densecap/` like below.
```
mkdir ROBOCUP_HOME/python_script/captioning/data/models/densecap/
cd ROBOCUP_HOME/python_script/captioning/data/models/densecap/
sftp -P3022 kimchi@147.46.219.78
get ~/Desktop/densecap-pretrained-vgg16.t7
exit
```

## Requirements
* UBUNTU 14.04 for ROS-INDIGO

Install below from their websites.
* NVIDIA driver 375.20
* CUDA 8.0
* CUDNN 5.1
* Tensorflow r1.1

Install below.

* Torch 
from http://torch.ch/docs/getting-started.html (choose 'yes' when installer asks something about path)

* SpeechRecognition 
    ```
    pip install SpeechRecognition
    ```

* ROS indigo desktop full 
from http://blog.naver.com/gliese581gg/220645607537 or http://wiki.ros.org/indigo/Installation/Ubuntu

* naoqi (add below to `~/.bashrc`)
    ```
    export PYTHONPATH=${PYTHONPATH}:~/'YOUR_PATH'/naoqi/lib/python2.7/site-packages
    ```

* sshpass 
    ```
    apt-get install sshpass
    ```

* others
    ```
    sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
    
    pip install Cython
    
    sudo apt-get install libhdf5-dev libblas-dev liblapack-dev gfortran
    
    pip install h5py
    
    pip install keras
    
    darkflow (https://github.com/thtrieu/darkflow) (intall option 3)
    
    apt-get install ros-indigo-navigation
    
    apt-get install ros-indigo-gmapping
    
    apt-get install ros-indigo-pepper-*
    
    luarocks install nn
    
    luarocks install image
    
    luarocks install lua-cjson
    
    luarocks install https://raw.githubusercontent.com/qassemoquab/stnbhwd/master/stnbhwd-scm-1.rockspec
    
    luarocks install https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/torch-rnn-scm-1.rockspec
    
    luarocks install cutorch
    
    luarocks install cunn
    
    luarocks install cudnn
    
    luarocks install md5
    
    luarocks install --server=http://luarocks.org/dev torch-ros
    
    pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl 
    
    pip install torchvision

    pip install nltk

    (open python and) import nltk ; nltk.download('punkt')

    pip install pattern

    pip install google-cloud
    
    ```

* tmux

    ```
    sudo apt-get update

    sudo apt-get install -y python-software-properties software-properties-common

    sudo add-apt-repository -y ppa:pi-rho/dev

    sudo apt-get update

    sudo apt-get install -y tmux=2.0-1~ppa1~t
    ```

## Catkin Compile 
merge the included `catkin_ws folder` to yours and `catkin_make`

    merge catkin_ws folder
    catkin_make


## How to use *PIO*

### Import *PIO* modules

```python
import pepper_io

pio = pepper_io.pepper_io()
```

* Note that only ***1*** pio instance should be made. Do not make multiple instances
* You must put your script in the included 'python_script' folder

### Before you run it
* Do every command with `sudo` or become `root`.

```
sudo -s
```

* Launch modules using Tmux

```

 # 1. launch modules
cd {path to robocup2017}/python_scripts
./run_pepper.sh {ip}

 #(example : ./run_pepper.sh 192.168.1.176)

 # 2. run your script in most right screen
python ~~.py --ip 192.168.1.~~

```

* Manually run each module

```
 #1. Start Driver
roslaunch pepper_jychoi pepper_start_jy.launch nao_ip:=192.168.1.~


 #2. Start navigation module
roslaunch pepper_jychoi pepper_navigation.launch map_file:=full/path/to/your_yaml_file

 #You will see 2D pose Estimate. Set robot location and move it with 2D Nav Goal to some place.
 #Pepper will find where it is.

 #3. run deep learning modules
cd {path to robocup2017}/python_scripts
python obj_detector.py --ip 192.168.1.~~
python reid_module.py
python pose_detector.py
cd captioning
th run_ros2.lua
``` 

### Object class
* See `objs.msg` to see which types are supported. For example,`int32 x` means x position.

    ```
    string class_string : object class
    int32 x : center of X of bounding box
    int32 y : center Y of bounding box
    int32 h : height of bounding box
    int32 w : width of bounding box
    float64 confidence : class score
    sensor_msgs/Image cropped : cropped image
    sensor_msgs/Image cropped_cloud : cropped point cloud
    geometry_msgs/Pose pose_wrt_robot : position of the object wrt the robot
    geometry_msgs/Pose pose_wrt_map : position of the object wrt the map
    geometry_msgs/Pose pose_wrt_odom : position of the object wrt the odometry
    int32 person_id : id of a person after person identification
    string person_name : (deprecated) name of a person
    float64 reid_score : confidence of re-identifiation of a person
    int32 isWaving : (deprecated) will be 1 if a person is waving his/her hands, 0 o.w.
    int32 isSitting : (deprecated) will be 1 if a person is sitting 
    int32 isRwaving : (deprecated) 1 if the person waving right hand
    int32 isLwaving : (deprecated) 1 if the person waving right hand
    int32 isLying : (deprecated) 1 if the person is lying
    int32 isLpointing : (deprecated) 1 if the person pointing right hand
    int32 isRpointing : (deprecated) 1 if the person pointing right hand
    string ucolor : (deprecated) color of the person's upper cloth
    string lcolor : (deprecated) color of the person's lower cloth
    float64[] joints : list of floats, represents the position of joints the indexes are for

    0 1 nose 2 3 neck 4 5 r_shoulder 6 7 r_elbow 8 9 r_wrist 10 11 l_shoulder 12 13 l_elbow 14 15 l_wrist 16 17 r_pelvis 18 19 r_knee 20 21 r_anckle 22 23 l_pervis 24 25 l_knee 26 27 l_ankle 28 29 r_eye 30 31 l_eye 32 33 r_ear 34 35 l_ear

    string[] captions : captions generated by captioning module
    string[] tags : contains useful tags for objects
 
    in current version, tags contain class of the object, 'waving','rwaving','lwaving','man','woman','sitting','lying','lpointing','rpointing','blue','green','red','white','black',person's name
    ```

    Deprecated variables are still working but I highly recommend to use the 'tags' instead of them

* See `obj_array.msg`.

    ```
    Header header : Contains timestamp and ETC, No need to modify
    objs[] objects : List of *objs.msg*
    int32 msg_idx : index of message (used for perception integration)
    sensor_msgs/Image scene_rgb : rgb image of whole scene
    sensor_msgs/Image scene_cloud : point cloud of whole scene
    ```

* And `pose` class which is returned from `pos_wrt_map` , `pose_wrt_robot` and `pose_wrt_odom`.
    ```
    # geometry_msgs/Pose
    Point position
    Quaternion orientation
    ```
    
    `position` consists of
    ```
    float64 x
    float64 y
    float64 z
    ```
    
    `orientaiton` consists of
    ```
    float64 x
    float64 y
    float64 z
    float64 w
    ```

### Example python script
```python
def main():
    pio = pepper_io(ip='192.168.1.176')
    test_objects = pio.get_perception()
    header = test_objects.header
    objets = test_objects.objects
    for obj in objects:
        print obj.class_string # print class of the object.
        pose = obj.pos_wrt_map # print position wrt map
        position_x = pose.position.x
        orientation_x = pose.orientation.x
        # cropped image is rosimg. you have to convert this to numpy img.
        cropped_image = pio.rosimg_to_numpyimg(obj.cropped) # this is numpy image
        tags = obj.tags
        
    return None
```


### Functions

#### get perception
```python
pio.get_perception(fil = None,reid=True,pose=True,captioning=True,time_limit = 3.0)
```
* `fil`: list of classes to find. find all classes if None (example : fil=['person','chair'])
* `reid`,`pose`,`captioning` : if True, output contains those information
* `time_limit` : return empty `obj_array` if latest information is older than this seconds

get integrated perception (objects, people's name, pose, captions)

returns `objs_array` instance.

#### get object information from individual modules (deprecated)
```python
pio.get_objects(fil = None,time_limit = 3.0)
pio.get_people_identified(waving_only=False,,time_limit = 3.0)
pio.get_people_wavings(name = None,time_limit = 3.0)
```
* `fil`: list of classes to find. find all classes if None (example : fil=['person','chair'])
* `waving_only` : if True, only waving people will be returned
* `name` : list of strings, if not None, only find specified people
* `time_limit` : return empty `obj_array` if latest information is older than this seconds

get object information from individual modules (object detection, identification, pose detection)

#### get captions of whole scene
```python
pio.get_captions()
```
get captions of whole scene. captions for people are automatically stored in `obj.captions` so don't use this to extract people's captions


#### save waypoints
```python
pio.save_waypoint(filename)
```
* `filename`: a file name without path. E.g.: `tour_guid.txt`

save waypoints to a file.


#### load waypoints
```python
pio.load_waypoints(filename)
```
* `filename`: a file name without path. E.g.: `tour_guid.txt`

reset current waypoints and load waypoints from a file.


#### add a waypoint
```python
pio.add_waypoint(name, location=None)
```
* `name`: `sring`
* `location`: a tuple, list or numpy array consts of three `float`s for `x`, `y`, `direction` in a map. The default value is the robot's location.

add a waypoint.

#### go to a waypoint
```python
pio.go_to_waypoint(name, wait=True, clear_costmap=False)
```
* `name`: `sring`
* `wait`: the function returns immediately if `False`. else blocked until the robot gets to the goal.
* `clear_costmap`: reset temporally detected obstacles if `True`, else use the previous costmap.

go to a waypoint.

#### get location
```python
pio.get_loc(p=np.array([0,0,0],o=np.array[0,0,0,0],source='CameraTopFrame',target='map')
```
* `p`: position tuple, list or numpy array consts of three `float`s for `x`, `y`, `z` wrt the robot. `z` must always be `0`.
* `o`: orientation tuple, list or numpy array consts of four `float`s for `x`, `y`, `z`, `w` wrt the robot. this is quaternion coordinate system
* `source`: source frame
* `target`: target frame

convert the location in source coordinate system to the location in target coordinate system

orientation must be quaternion (see also `pio.yaw_to_quat` and `pio.quat_to_yaw`)



#### global localization
```python
pio.global_localization()
```

stabilize the robot's position by rotating several times.


#### automatic speech recognition
```python
pio.init_speech_recognition(sensitivity=0.3)
```
* `sensitivity`: `0` ~ `1`

initialize speech recognition. Afther this, speeches are detected countinousely.

See also
```python
pio.set_sound_sensitivity(sensitivity=0.9)
```

#### manual speech recognition
```python
pio.start_recording(reset=False, base_duration=3.0)
```
* `reset`: start a new recoding session for `True`, or add to an existing one if `False`
* 'base_duration': minimum length of duration. Only for `reset` is `True`.

Detect a speech from now manually.


#### get rgb
```python
pio.get_rgb()
```

Returns camera image to numpy array.

See also `pio.get_depth()` and `pio.get_point_cloud`.

The size of image is always **320(width) * 240(height)**.

#### get depth
```python
pio.get_depth()
```

returns depth image numpy array

#### get point cloud
```python
pio.get_point_cloud()
```

returns point cloud numpy array


#### get position from point cloud
```python
pio.get_pos_wrt_robot(x,y,size=10,scan_len=50,scan='point')
```
* `x`, `y`: position of a pixel of interest
* `size` : if `scan` is 'point', the function returns nearest point in surrounding `size` area of `x`,`y`
* `scan_len` : if `scan` is 'line', the function returns nearest point in vertical line from `x`,`y` to `x+scan_len`,`y`
* `scan` : 'point' or 'line'

Get a position of specific pixel in meters.

scan surrounding area or vertical line.


#### animate a gesture
```python
pio.do_anim(command)
 #example : pio.do_anim('Gestures/Hey_1')
```
* `command`: a name of the gesture in `string`.

Animate a gesture. The kinds of gestures are from http://doc.aldebaran.com/2-4/naoqi/motion/alanimationplayer-advanced.html#alanimationplayer-advanced.



#### change the posture of the robot
```python
pio.do_pose(command)
```
* `command`: a name of the posture in `string` among Crouch, LyingBack, LyingBelly, Sit, SitRelax, Stand, StandInit, StandZero.

Change the posture of the robot.


#### move to a goal
```python
pio.go_to_goal(x, y, theta, wait=True, clear_costmap=False)
```
* `x`, `y`, `theta`: the position of the goal wrt the map. `theta` is in radian.
* `wait`: the function returns immediately if `False`. else blocked until the robot gets to the goal.
* `clear_costmap`: reset temporally detected obstacles if `True`, else use the previous costmap.

Move the robot to a location.


#### set velocity
```python
pio.set_velocity(x y, theta, duration=-1.)
```
* `x`, `y`: the velocity of forward-backward diretion and left-right direction.
* `theta`: the angular velocity of rotation.
* `duration`: the duration of the movement. ***NOTE: if not specified, the robot moves forever in the specified direction. Colision!***

Move the robot in a speciiec direction for a duration and velocity.


#### stop
```python
pio.stop()
```

Stop ongoing movement.


#### keyboard control
```python
pio.activate_keyboard_contol()
```

Control the robot using a keyboard.

```
usage : type following command and press enter.
		robot will maintain velocity unless you give another command.
commands:
	 w : forward
	 s : stop
	 x : backward
	 a : strafe left
	 d : strafe right
	 q : turn left
	 e : turn right
	 say : say next input
	 waypoint : add current location to waypoints
	 save_waypoint : save all waypoints
	 add_waypoint : add current location to waypoints
	 go_waypoint : go to waypoint
	 add_reid_target : add reid target
	 save_reid_targets : save reid target
	 sr : speech recognition
	 c : exit
```


#### say something
```python
pio.say(text)
```
* `text`: `string` to say.

Say a text. Speech recognition is paused during saying.



#### convert image
```python
pio.rosimg_to_numpyimg(img_msg)
```
* `img_msg`: ros image to convert.

Convert ros image to numpy image.

See also
```python
pio.numpyimg_to_rosimg(npimg)
```

#### find a word
```python
pio.find_ord(word, source=None)
```
* `word`: a word to find.
* `source`: the source of speech to find. Use the latest one if not specified.

Find a word from a recorded speech. 

#### follow a person
```python
pio.follow_person(target, target_dist=1.0, timeout=60, stop_criterion='dist', use_reid=False, reid_name=None, stop_word='stop')
```
* `target`: a target to follow in `objs` instance.
* `target_dist`: distance in meters.
* `timeout`: fails if the stop criterion is not satisfied until `timeout`.
* `stop_criterion`: `'dist'` for distance or `'speech'` for a word.
* `use_reid`: follow the closest person in location if `False`, follow the identified person if `True`.
이전 로케이션에서 가장 가까운 사람을 찾기 if `False`, 아이덴티피케이션 결과가 일치하는 사람 따라가기 if `True`.
* `reid_name`: follow the person whoes name is `reid_name`, else name the person as `follow`.
* `stop_word`: the word to make the robot stop following if `stop_criterion` is `'speech'`.

Follow a person until the stop criterion is satisfied.


### Attribute 
#### pio.speech_memory
the most recently recognized speech in the `string`.

#### pio.speech_hints
A list of strings used for speech recognition as hints.
