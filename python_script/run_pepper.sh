#! /bin/bash

#tmux set -g mode-mouse on
#tmux set -g mouse-resize-pane on
#tmux set -g mouse-resize-pane on
#tmux set -g mouse-select-window on

tmux new-session -d
#tmux new-window
tmux split-window -h
tmux select-pane -L
tmux split-window -v
tmux split-window -h
tmux split-window -v
tmux select-pane -U
tmux select-pane -U
tmux split-window -h
tmux split-window -v
#tmux attach-session -d

#tmux select-pane -L -R -U -D or -t 0~6

tmux select-pane -t 0
tmux send "source ~/.bashrc" C-m
tmux send "roslaunch pal_pepper pepper_start_jy.launch nao_ip:=$1" C-m
#tmux rename-window "ROS-Main"

echo "Waiting..."
sleep 7

tmux select-pane -t 3
tmux send "source ~/.bashrc" C-m
tmux send "roslaunch pal_pepper pepper_navigation.launch" C-m
#tmux rename-window "ROS-Navigation"

tmux select-pane -t 1
tmux send "source ~/.bashrc" C-m
tmux send "python obj_detector.py --ip $1" C-m

tmux select-pane -D
tmux send "source ~/.bashrc" C-m
tmux send "python pose_detector.py" C-m

tmux select-pane -D
tmux send "source ~/.bashrc" C-m
tmux send "cd captioning" C-m
tmux send "th run_ros2.lua" C-m

tmux select-pane -D
tmux send "source ~/.bashrc" C-m
tmux send "python reid_model.py" C-m

tmux select-pane -R
tmux send "source ~/.bashrc" C-m

tmux attach-session -d
