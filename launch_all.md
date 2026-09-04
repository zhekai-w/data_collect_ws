### Setup localhost in container for zed to connect
```
sudo apt install openssh-server
sudo service ssh start
sudo passwd $USER

tailscale up --accept-dns=false
```

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
conda activate lerobot
source ./install/setup.bash
export PYTHONPATH=/home/zack/miniconda3/envs/lerobot/lib/python3.10/site-packages:/home/zack/work/pkgs/lerobot/src:$PYTHONPATH
ros2 run ur5_lerobot_data_collection data_collect --ros-args -p task:="place apple in the wooden plate."
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
    --batch_size= 32 \
    --gradient_accumulation_steps=4 \
    --data-config="ur5_2f85_arm_gripper" 
```

### GR00T eval with dataset
```
python scripts/eval_policy.py --plot \
  --model-path $HOME/work/models/gr00t_finetuned/gr00t_81x9D_E32B_R32_fruit \
  --dataset-path $HOME/work/all_datasets/2_std_datasets/test_encoded \
  --embodiment-tag new_embodiment \
  --data-config ur5_2f85_arm_gripper \
  --modality-keys ur5_arm gripper \
  --start_traj=1 --steps=120 \
  --video-backend decord \
  --denoising-steps=8 \
  --action_horizon=16 --chunk-filter rts --boundary-blend

# Visualize savgol smoothed vs raw
python eval_policy.py --plot --chunk-filter savgol

# Visualize with boundary blend
python eval_policy.py --plot --chunk-filter savgol --boundary-blend

# Visualize RTS Kalman smoother
python eval_policy.py --plot --chunk-filter rts --boundary-blend

```

### GR00T eval on hardware with dataset
#### Launch ur5 robot driver
```
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.1.100 \
initial_joint_controller:=forward_position_controller \
launch_dashboard_client:=false
```
#### Launch gripper controller 
```
ros2 launch robotiq_description robotiq_control.launch.py
```
#### Run eval script
```
python scripts/eval_policy_hardware.py \
  --model-path $HOME/work/models/gr00t_finetuned/gr00t_64x9D_E128B_fruit_shift \
  --dataset-path $HOME/work/all_datasets/1_std_datasets/test_fruit_encoded_shift \
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
python scripts/inference_service.py --model-path $HOME/work/models/gr00t/gr00t_81x9D_E128B_fruit --server \
--data-config ur5_2f85_arm_gripper \
--embodiment-tag new_embodiment 

python scripts/inference_service.py --model-path $HOME/work/models/gr00tsmolvla/gr00tsmolvla_81x9D_E64B_fruit --server \
--data-config ur5_gr00tsmolvla \
--embodiment-tag new_embodiment 
```

#### launch ur_robot_driver
#### Pick ONE initial_joint_controller — it maps to the client mode:
####   scaled_joint_trajectory_controller → streaming client DEFAULT (jtc_topic): C++ 125Hz spline, no GIL jitter
####                                        ALSO simple_client and timed_chunk_client (action interface)
####   forward_position_controller        → streaming client --control-interface forward_position: Python-generated
####                                        dense setpoint stream at --stream-hz, exact VLA waypoints
#### ur5_real_controllers.yaml sets open_loop_control: true on the scaled JTC — REQUIRES the patched
#### ur_controllers in src/Universal_Robots_ROS2_Driver (stock 2.13.0 throws "can't compare times with
#### different time sources" on activation). Source install/setup.bash. This is what removed the jitter.
#### Rebuild dualsense_teleop after editing the yaml — the launch reads the INSTALLED copy.

# jtc_topic mode (streaming client default, simple_client, timed_chunk_client):
```
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.1.100 \
runtime_config_package:=dualsense_teleop controllers_file:=ur5_real_controllers.yaml \
initial_joint_controller:=scaled_joint_trajectory_controller \
launch_dashboard_client:=false
```

# fpc mode (streaming client --control-interface forward_position):
```
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.1.100 \
runtime_config_package:=dualsense_teleop controllers_file:=ur5_real_controllers.yaml \
initial_joint_controller:=forward_position_controller
```

#### fpc mode only: patched description with tunable servoj_gain / servoj_lookahead_time.
#### Irrelevant to jtc_topic (servoj params apply to forward_position_controller only).
#### tune: edit src/dualsense_teleop/urdf/ur.ros2_control.xacro ("TUNE HERE"), colcon build --packages-select dualsense_teleop, restart driver
```
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.1.100 \
description_package:=dualsense_teleop description_file:=ur_patched.urdf.xacro \
runtime_config_package:=dualsense_teleop controllers_file:=ur5_real_controllers.yaml \
initial_joint_controller:=forward_position_controller
```
#### Launch gripper controller 
```
ros2 launch robotiq_description robotiq_control.launch.py
```
#### Run inference simple client
```
# Most smoothing (Kalman)
python scripts/ur5_gr00t_simple_client.py --send-mode chunk --lang "place apple in the basket." --chunk-filter rts --boundary-blend

# Use with forward position controller
python ur5_gr00t_simple_client.py --controller forward_position_controller --send_mode chunk
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

python scripts/visualize_text_image_cross_attention.py \
  --image_path scripts/images/fruit1.png --text_query mango \
  --box_source gdino --box_model IDEA-Research/grounding-dino-base \
  --box_negatives 'lemon,banana,apple,green pepper,yellow pepper' \
  --box_threshold 0.15
```

### Fine-tuning smolVLA script
```
python3 lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=ur5_combined \
  --dataset.root=/home/zack/work/all_datasets/3_std_datasets/3_combined_encoded \
  --batch_size=64 \
  --steps=100000 \
  --save_freq=10000 \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --output_dir=/home/toastoast/zack_ws/models/smolvla_ur5_simpletask \
  --num_workers=8 
```

### SmolVLA eval
```
python3 ./src/lerobot/scripts/eval_lerobot_policies.py \
    --policy_path /home/zack/work/models/smolvla/smolvla_ur5_simpletask/checkpoints/100000/pretrained_model \
    --dataset_root /home/zack/work/all_datasets/3_std_datasets/apple_to_basket_encoded \
    --episodes 1 \
    --steps 120 \
    --chunk_filter rts --boundary_blend
```

### SmolVLA inference
```
python3 src/lerobot/scripts/server/smolvla_inference_service.py \
--server --model-path /home/zack/work/models/smolvla/smolvla_ur5_fruit/checkpoints/100000/pretrained_model --port 5555


python scripts/ur5_gr00t_simple_client.py --send-mode chunk \
    --chunk-filter rts --boundary-blend --action_horizon 50 \
    --lang "place apple in the basket." 
```

### Openpi eval
```
uv run python scripts/eval_policy.py \
    --config pi0_ur5_lora \
    --checkpoint-dir /home/zack/work/models/pi0_ur5_lora/pi0_ur5_lora/29999 \
    --dataset-path /home/zack/work/all_datasets/2_std_datasets/test_encoded \
    --trajs 5 \
    --plot
```

### Openpi inference
```
uv run python scripts/openpi_inference_service.py --server --config pi0_ur5_lora \
    --checkpoint-dir /home/zack/gr00t_lerobot_ws/models/pi0_ur5_lora_D243/59999 \
    --port 5555
    
# Smoke test (separate terminal, no robot)
python pkgs/openpi/scripts/openpi_inference_service.py --client --port 5555
```

# Production client
python scripts/ur5_gr00t_simple_client.py --send-mode chunk \
  --lang "place apple in the basket." \
  --host 163.13.164.145 --port 5555 \
  --chunk-filter rts --boundary-blend \
  --action_horizon 30 --dt 0.07

# PRODUCTION. jtc_topic / ramp / threshold 0.2 / lookahead 30 / end-velocity
# continue are the DEFAULTS now — this is the whole command.
python scripts/ur5_gr00t_streaming_client.py \
  --lang "place apple in the basket." \
  --action_horizon 50 \
  --dt 0.05

# Add --log to write streaming_log.npz for scripts/compare_logs.py.
# A/B against the old behavior: --aggregate_fn_name weighted_average,
# --no-jtc-absolute-timing, --tick-slack 0, --control_interface forward_position.

python scripts/ur5_gr00t_timed_chunk_client.py \
  --lang "place apple in the basket." \
  --host 163.13.164.145 --port 5555 \
  --chunk-filter rts \
  --aggregate_fn_name ramp \
  --action_horizon 30 --dt 0.07

python scripts/plot_streaming_log.py streaming_log.npz
```

#### streaming client: setpoint smoothing A/B (forward_position only)
Defaults are now C1 cubic Hermite between waypoints + a 2nd-order Butterworth on
the dense 125 Hz setpoint stream (the moveit_servo ButterworthFilterPlugin
position — filter *after* the interpolator, not at waypoint rate). Both are
flag-switchable, so before/after needs no rebuild:
```
--segment-interp linear  --stream-filter none                     # old behavior, exact
--segment-interp hermite --stream-filter none                     # interpolation only
--segment-interp hermite --stream-filter butter --stream-filter-cutoff 8   # default
--segment-interp linear  --stream-filter butter --stream-filter-cutoff 8   # filter only
```
`--filter` (sparse OneEuro at 1/dt) is auto-disabled when a stream filter is on.
`plot_streaming_log.py` prints RMS/max jerk, RMS acc and max|Δv| for the dense
stream and writes `*_stream.png`. Background + results table:
`streaming_client_slowness_notes.md`.

### Openpi convert jax to pytorch
```
uv run python examples/convert_jax_model_to_pytorch.py \
    --checkpoint-dir /home/zack/work/models/pi0_ur5_lora/pi0_ur5_lora/29999 \
    --config-name pi0_ur5_lora \
    --output-path /home/zack/work/models/pi0_ur5_lora/pytorch_29999 \
    --precision bfloat16

# Copy norm stats (conversion script looks in wrong place for step-dir checkpoints)
cp -r /home/zack/work/models/pi0_ur5_lora/pi0_ur5_lora/29999/assets \
       /home/zack/work/models/pi0_ur5_lora/pytorch_29999/
```
