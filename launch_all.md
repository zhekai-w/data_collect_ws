## launch all
### ur5 controller 
```
ros2 launch dualsense_teleop ur5_pose_tracking.launch.py
```
### gripper controller
```
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
ros2 run ur5_lerobot_data_collection data_collect --ros-args -p task:="place the small cube on the red box."
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
    --lora-rank=16 \
    --gradient_accumulation_steps=4 \
    --data-config="ur5_2f85_arm_gripper" 
```

### GR00T server
```
PYTHONNOUSERSITE=1 python scripts/inference_service.py --server \
  --model-path /home/zhekai/work/pkgs/gr00t_finetuned/gr00t_32x4batch_30ksteps \
  --embodiment-tag new_embodiment \
  --data-config ur5_2f85_arm_gripper
```

### GR00T eval with dataset
```
python scripts/eval_policy.py --plot \
  --model-path $HOME/work/pkgs/gr00t_finetuned/gr00t_32x4batch_30ksteps \
  --dataset-path  $HOME/work/dataset \
  --embodiment-tag new_embodiment \
  --data-config ur5_2f85_arm_gripper \
  --modality-keys ur5_arm gripper \
  --start_traj=45 --steps=300 --filter \
  --video-backend decord \
  --denoising-steps=8 \
  --action_horizon=4 \
```

### GR00T eval on hardware with dataset
#### Launch ur5 robot driver
```
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.1.100 \
initial_joint_contoller:=forward_position_controller 
```
#### Launch gripper controller 
```
ros2 launch robotiq_description robotiq_control.launch.py
```
#### Run eval script
```
python scripts/eval_policy_hardware.py \
  --model-path $HOME/work/pkgs/gr00t_finetuned/gr00t_32x4batch_30ksteps \
  --dataset-path  $HOME/work/dataset \
  --embodiment-tag new_embodiment \
  --data-config ur5_2f85_arm_gripper \
  --modality-keys ur5_arm gripper \
  --start_traj=45 --steps=300 --filter --send-mode chunk \
  --denoising-steps=8 \
  --action_horizon=4 \
  --dt=0.033 
```

### GR00T inference
#### run inferece server
```

```

#### launch ur_robot_driver 
#### scaled_joint_trajectory_controller for full chunk execution
#### forward_position_controller for one action per tick control
```
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.1.100 \
initial_joint_contoller:=scaled_joint_trajectory_contoller 
```



