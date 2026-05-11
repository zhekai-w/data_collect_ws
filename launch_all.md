## launch all
### ur5 controller 
```
rosdep install --from-paths src --ignore-src -r -y

ros2 launch dualsense_teleop ur5_pose_tracking.launch.py
```
### gripper controller
```
sudo chmod 777 /dev/ttyUSB0 
ros2 launch robotiq_description robotiq_control.launch.py
```

### dualsense teleoperation
```
might need to "colcon build --symlink-install --packages-select dualsense_teleop" the first time running container
pip install scipy==1.15.3
sudo chmod 777 /dev/ttyUSB0 
ros2 launch dualsense_teleop dualsense_teleop.launch.py
```

### LeRobot data collection 
```
ros2 run ur5_lerobot_data_collection data_collect --ros-args -p task:="place apple in the white plate."
```

### Replay collected data
```
ros2 run ur5_lerobot_data_collection data_replay \
  --dataset ./All_Datasets/dataset_small_to_orange --episode 0 --speed 0.5 --mode trajectory
```

### UR5 arm home pose
- Translation: [-0.044, 0.426, 0.427]
- Rotation: in Quaternion (xyzw) [1.000, -0.015, 0.007, -0.017]
- Rotation: in RPY (radian) [-3.107, -0.014, -0.029]
- Rotation: in RPY (degree) [-178.040, -0.815, -1.688]
- Matrix:
  0.999 -0.029  0.015 -0.044
 -0.029 -0.999  0.034  0.426
  0.014 -0.034 -0.999  0.427
  0.000  0.000  0.000  1.000

### GR00T fine-tuning
#### Write your datasets as list in bash before running training script
```
dataset_list=(
    "$HOME/zack_ws/lerobot_datasets/All_Datasets/dataset_large_to_green"
    "$HOME/zack_ws/lerobot_datasets/All_Datasets/dataset_large_to_orange"
    "$HOME/zack_ws/lerobot_datasets/All_Datasets/dataset_large_to_red"
    "$HOME/zack_ws/lerobot_datasets/All_Datasets/dataset_medium_to_green"
    "$HOME/zack_ws/lerobot_datasets/All_Datasets/dataset_medium_to_orange"
    "$HOME/zack_ws/lerobot_datasets/All_Datasets/dataset_medium_to_red"
    "$HOME/zack_ws/lerobot_datasets/All_Datasets/dataset_small_to_green"
    "$HOME/zack_ws/lerobot_datasets/All_Datasets/dataset_small_to_orange"
    "$HOME/zack_ws/lerobot_datasets/All_Datasets/dataset_small_to_red"
)
```
#### Then run training script
```
python3 scripts/gr00t_finetune.py \
    --dataset-path ${dataset_list[@]} \
    --max_steps=30000 \
    --save_steps=10000 \
    --lora-rank=16 \
    --gradient_accumulation_steps=4 \
    --data-config="ur5_2f85_arm_gripper" 
```

### GR00T eval with dataset
```
python scripts/eval_policy.py --plot \
  --model-path $HOME/work/Models/gr00t_finetuned/gr00t_mediumD_32B_30Hz \
  --dataset-path $HOME/work/All_Datasets/test_just_medium \
  --embodiment-tag new_embodiment \
  --data-config ur5_2f85_arm_gripper \
  --modality-keys ur5_arm gripper \
  --start_traj=1 --steps=120 \
  --video-backend decord \
  --denoising-steps=8 \
  --action_horizon=16 --filter 
```

### GR00T eval on hardware with dataset
#### Launch ur5 robot driver
```
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.1.100 \
initial_joint_controller:=passthrough_trajectory_controller
initial_joint_controller:=forward_position_controller 
```
#### Launch gripper controller 
```
ros2 launch robotiq_description robotiq_control.launch.py
```
#### Run eval script
```
python scripts/eval_policy_hardware.py \
  --model-path $HOME/work/Models/gr00t_finetuned/gr00t_64x9D_32x4B_30kS \
  --dataset-path $HOME/work/All_Datasets/test_datasets \
  --embodiment-tag new_embodiment \
  --data-config ur5_2f85_arm_gripper \
  --modality-keys ur5_arm gripper \
  --start_traj=1 --steps=300 --filter --send-mode single \
  --denoising-steps=8 \
  --action_horizon=16 \
  --dt=0.3
```

### GR00T inference
#### run inferece server
```
python scripts/inference_service.py --model-path $HOME/work/Models/gr00t_finetuned/gr00t_mediumD_32B_30Hz --server \
--data-config ur5_2f85_arm_gripper \
--embodiment-tag new_embodiment 

```

#### launch ur_robot_driver 
#### scaled_joint_trajectory_controller for full chunk execution
#### forward_position_controller for one action per tick control
```
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.1.100 \
initial_joint_contoller:=scaled_joint_trajectory_contoller 
```
#### Launch gripper controller 
```
ros2 launch robotiq_description robotiq_control.launch.py
```
#### Run inference client
```
python scripts/ur5_gr00t_simple_client.py --send-mode chunk --lang "place the cube on the red box." 
```

#### Replay collected data
```
python data_replay.py --dataset-path /home/zack/work/All_Datasets/dataset_small_to_orange --controller passthrough \
--send-mode single
```

#### Visualize attention head
```
python scripts/visualize_text_image_cross_attention.py --image_path ./scripts/images/Balls.jpg --text_query "white ball" --head 5 --layers 16

python scripts/visualize_text_image_cross_attention.py --image_path ~/work/All_Datasets/30hz/large_to_red_encoded/videos/chunk-000/observation.images.cam1/episode_000003.mp4 --text_query "cube" --frame_stride 5 --head 11
```

### Fine-tuning smolVLA script
```
python3 lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=ur5_combined \
  --dataset.root=/home/toastoast/zack_ws/lerobot_datasets/30hz/reordered_combined_nodepth \
  --batch_size=64 \
  --steps=100000 \
  --save_freq=10000 \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --output_dir=/home/toastoast/zack_ws/models/smolvla_ur5 \
  --num_workers=8
```