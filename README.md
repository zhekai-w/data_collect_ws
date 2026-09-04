# gr00t_lerobot_ws

ROS 2 Humble workspace for collecting UR5 + Robotiq 2F-85 demonstrations with a
DualSense controller, fine-tuning vision-language-action (VLA) policies on them,
and deploying the policies back to the real robot.

Four policy families are supported. All of them serve behind the same ZMQ wire
protocol, so one set of robot clients works with every model.

| Model | Package | Fine-tune | Serve |
|---|---|---|---|
| GR00T N1.5 (flow-matching DiT head) | `pkgs/Isaac-GR00T` | `scripts/gr00t_finetune.py` | `scripts/inference_service.py` |
| GR00T N1.5 + SmolVLA action expert | `pkgs/Isaac-GR00T` | `scripts/gr00t_finetune_smolvla.py` | `scripts/inference_service.py` |
| SmolVLA | `pkgs/lerobot` | `lerobot/scripts/train.py` | `lerobot/scripts/server/smolvla_inference_service.py` |
| pi0 / pi0-FAST (openpi) | `pkgs/openpi` | `scripts/train.py` | `scripts/openpi_inference_service.py` |

---

## Getting the code

The workspace is one git repository with eight submodules, most of them forks.
Clone it recursively:

```bash
git clone --recurse-submodules https://github.com/zhekai-w/data_collect_ws.git gr00t_lerobot_ws
cd gr00t_lerobot_ws
```

If it was already cloned without submodules:

```bash
git submodule update --init --recursive
```

The clone directory can be named anything, but keep a `_ws` suffix. The Docker
helper scripts in `ubuntu22.04_ros2/` derive the image and container name from
it (`get_param.sh`).

Submodules and the branch each one tracks:

| Path | Repository | Branch |
|---|---|---|
| `src/dualsense_teleop` | zhekai-w/dualsense_teleop | default |
| `src/ur5_lerobot_data_collection` | zhekai-w/ur5_lerobot_data_collection | default |
| `src/ros2_robotiq_gripper` | zhekai-w/ros2_robotiq_gripper | `humble` |
| `src/serial` | tylerjw/serial (upstream) | `ros2` |
| `src/Universal_Robots_ROS2_Driver` | zhekai-w/Universal_Robots_ROS2_Driver | `my-humble` |
| `pkgs/Isaac-GR00T` | zhekai-w/Isaac-GR00T | `my-n1.5-release` |
| `pkgs/lerobot` | zhekai-w/lerobot | `my-v0.3.3` |
| `pkgs/openpi` | zhekai-w/openpi | default |

A fresh clone checks each submodule out at the commit the parent repository
pins, in detached HEAD state. To work on one, check out its branch first:

```bash
cd pkgs/Isaac-GR00T
git checkout my-n1.5-release
```

Pull later changes for everything at once:

```bash
git pull
git submodule update --init --recursive
```

After committing inside a submodule, push it, then record the new pointer in
the parent repository:

```bash
cd pkgs/Isaac-GR00T && git push && cd ../..
git add pkgs/Isaac-GR00T
git commit -m "bump Isaac-GR00T"
```

Datasets (`all_datasets/`), checkpoints (`models/`), and colcon output
(`build/`, `install/`, `log/`) are gitignored and not part of the clone.

---

## 1. Layout

```
gr00t_lerobot_ws/
├── src/                              ROS 2 packages (colcon)
│   ├── dualsense_teleop/             DualSense IMU teleop, pose tracking, gripper teleop,
│   │                                 controller yaml + patched URDF for the real UR5
│   ├── ur_pose_tracking/             C++ Cartesian pose-tracking node (MoveIt Servo based)
│   ├── ur5_lerobot_data_collection/  LeRobot dataset recorder, replay, camera publishers,
│   │                                 scripts/ for encoding, merging, action shift, joint reorder
│   ├── ros2_robotiq_gripper/         Robotiq 2F-85 driver (PickNik, humble)
│   ├── serial/                       serial lib for the gripper
│   └── Universal_Robots_ROS2_Driver/ fork of UR driver 2.13.0; ONLY ur_controllers is built
├── pkgs/
│   ├── Isaac-GR00T/                  GR00T N1.5 fork + UR5 clients (branch my-n1.5-release)
│   ├── lerobot/                      LeRobot fork with SmolVLA server (branch my-v0.3.3)
│   └── openpi/                       openpi fork with UR5 config + ZMQ server
├── all_datasets/                     LeRobot v2.1 datasets + preprocessing scripts
├── models/                           fine-tuned checkpoints (gitignored)
├── ubuntu22.04_ros2/                 Docker image with ROS 2 Humble + 3 conda envs
├── project_notes/                    design notes (filters, streaming, attention, ...)
└── launch_all.md                     raw command scratchpad this README is based on
```

Every directory under `src/` and `pkgs/` is a git submodule except
`src/ur_pose_tracking`, which is tracked directly (see
[Getting the code](#getting-the-code)).

---

## 2. Hardware

| Device | Interface | Notes |
|---|---|---|
| UR5 (CB3) | Ethernet, robot at `192.168.1.100`, PC at `192.168.1.199` | External Control URCap must be running on the pendant |
| Robotiq 2F-85 | USB serial `/dev/ttyUSB0` | `sudo chmod 777 /dev/ttyUSB0` before launch |
| Azure Kinect | USB 3 | dataset `cam1`, model key `video.azure_kinect` |
| Intel RealSense | USB 3 | dataset `cam2`, model key `video.realsense` / `video.wfov` |
| Sony DualSense | USB or Bluetooth | udev rules in `setup_step.txt` |

Robot state vector is 7-D: six UR5 joints in `[shoulder_pan, shoulder_lift,
elbow, wrist_1, wrist_2, wrist_3]` order plus one gripper value. Actions have
the same layout. Datasets are recorded at 30 Hz.

UR5 arm home pose (TCP in base frame):

```
translation  [-0.044, 0.426, 0.427]
quat (xyzw)  [1.000, -0.015, 0.007, -0.017]
rpy (deg)    [-178.04, -0.82, -1.69]
```

---

## 3. Environment setup

### 3.1 Docker

`ubuntu22.04_ros2/lerobot.Dockerfile` builds an image with CUDA 12.8, ROS 2
Humble desktop, MoveIt, `ros-humble-ur`, Azure Kinect SDK, librealsense, and
three conda environments:

| conda env | Python | Purpose |
|---|---|---|
| `gr00t_n1.5` | 3.10 | GR00T fine-tune, eval, inference server, UR5 clients |
| `lerobot` | 3.10 | data collection node, SmolVLA train/eval/serve |
| `openpi` | 3.11 | pi0 train/eval/serve (uses `uv` inside the env) |

```bash
cd ubuntu22.04_ros2
./build_lerobot_env.sh        # build image <dockerhub-user>/lerobot_data_collect
./run_lerobot_env.sh          # run it: --network=host, --privileged, /dev and X11 mounted
```

The workspace is mounted at `/home/$USER/work` inside the container. All
`$HOME/work/...` paths below refer to that mount.

Optional, for connecting an editor (Zed) over SSH into the container:

```bash
sudo apt install openssh-server
sudo service ssh start
sudo passwd $USER
tailscale up --accept-dns=false
```

### 3.2 ROS 2 workspace build

Inside the container:

```bash
cd ~/work
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
echo "export ROS_DOMAIN_ID=69" >> ~/.bashrc
```

Notes:

- `src/Universal_Robots_ROS2_Driver` has `COLCON_IGNORE` in every package
  except `ur_controllers`. The driver itself comes from apt
  (`ros-humble-ur`); only the patched `ur_controllers` is overlaid. See
  the fork notice in its README for what the patch does and why.
- `pip install dualsense-controller scipy==1.15.3` and
  `sudo apt-get install ros-humble-imu-tools libhidapi-dev` are needed for
  the DualSense nodes (see `setup_step.txt`).
- The UR launch reads the INSTALLED copy of `ur5_real_controllers.yaml`.
  After editing anything under `src/dualsense_teleop/config` or `urdf`, run
  `colcon build --packages-select dualsense_teleop` and restart the driver.

### 3.3 Python environments

Each `pkgs/*` submodule is installed editable into its conda env by the
Dockerfile. For the `lerobot` env used together with ROS nodes, the ROS
Python path must be extended:

```bash
conda activate lerobot
source ./install/setup.bash
export PYTHONPATH=$HOME/miniconda3/envs/lerobot/lib/python3.10/site-packages:$HOME/work/pkgs/lerobot/src:$PYTHONPATH
```

---

## 4. Bring-up (robot, gripper, cameras)

Each block is one terminal. Source `install/setup.bash` in every terminal.

### 4.1 UR5 driver

The choice of `initial_joint_controller` decides which client mode works
later (section 9.3). Both variants use the workspace controller yaml.

**Scaled joint trajectory controller** (default for the streaming client,
also used by the simple and timed-chunk clients):

```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.1.100 \
  runtime_config_package:=dualsense_teleop controllers_file:=ur5_real_controllers.yaml \
  initial_joint_controller:=scaled_joint_trajectory_controller \
  launch_dashboard_client:=false
```

`ur5_real_controllers.yaml` sets `open_loop_control: true` on this
controller. Stock `ur_controllers` 2.13.0 throws
`can't compare times with different time sources` on activation with that
setting. The fork in `src/Universal_Robots_ROS2_Driver` fixes it. This is
what removed the inter-chunk jitter.

**Forward position controller** (dense setpoint streaming from Python):

```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.1.100 \
  runtime_config_package:=dualsense_teleop controllers_file:=ur5_real_controllers.yaml \
  initial_joint_controller:=forward_position_controller
```

Optional for forward-position only: a patched URDF exposing `servoj_gain` and
`servoj_lookahead_time`. Tune in `src/dualsense_teleop/urdf/ur.ros2_control.xacro`
("TUNE HERE"), rebuild `dualsense_teleop`, restart the driver.

```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.1.100 \
  description_package:=dualsense_teleop description_file:=ur_patched.urdf.xacro \
  runtime_config_package:=dualsense_teleop controllers_file:=ur5_real_controllers.yaml \
  initial_joint_controller:=forward_position_controller
```

Testing without a robot:

```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=0.0.0.0 \
  use_fake_hardware:=true launch_rviz:=true
```

### 4.2 Gripper

```bash
sudo chmod 777 /dev/ttyUSB0
ros2 launch robotiq_description robotiq_control.launch.py
```

Manual test:

```bash
ros2 action send_goal /robotiq_gripper_controller/gripper_cmd \
  control_msgs/action/GripperCommand "{command: {position: 0.0, max_effort: 50.0}}"
```

### 4.3 Cameras

The data collection node and the inference clients open the Azure Kinect
directly through `pyk4a` (1080p, 30 fps, depth off) and the RealSense through
`pyrealsense2`. No separate camera node is needed for those flows.

For visualising or logging camera streams in ROS, publishers exist:

```bash
ros2 run ur5_lerobot_data_collection all_cam_pub     # /camera/{azure,realsense}/{color,depth}
ros2 run ur5_lerobot_data_collection camera_sub      # display
```

---

## 5. Teleoperation and data collection

### 5.1 Cartesian pose tracking controller

Runs `ur_pose_tracking` (MoveIt Servo based) on top of the UR driver so the
DualSense can drive the TCP.

```bash
ros2 launch dualsense_teleop ur5_pose_tracking.launch.py
```

### 5.2 DualSense teleop

Launches the joystick publisher, IMU publisher, Madgwick filter, orientation
offset, gripper teleop, and the pose-tracking bridge.

```bash
sudo chmod 777 /dev/ttyUSB0
ros2 launch dualsense_teleop dualsense_teleop.launch.py
```

First run in a fresh container may need
`colcon build --symlink-install --packages-select dualsense_teleop`.
Details of the IMU pipeline are in `project_notes/dualsense_madgwick_moveit_servo.md`.

### 5.3 Record a LeRobot dataset

```bash
conda activate lerobot
source ./install/setup.bash
export PYTHONPATH=$HOME/miniconda3/envs/lerobot/lib/python3.10/site-packages:$HOME/work/pkgs/lerobot/src:$PYTHONPATH
ros2 run ur5_lerobot_data_collection data_collect --ros-args -p task:="place apple in the wooden plate."
```

The node subscribes to `/joint_states`, `/gripper/joint_states`,
`/gripper/commands` and `/forward_position_controller/commands`, captures
both cameras, and writes a LeRobot v2.1 dataset with features
`observation.state (7)`, `action (7)`, `observation.images.cam1`,
`observation.images.cam2` at 30 Hz. The `task` parameter becomes the
language instruction.

### 5.4 Replay a recorded episode

Sends recorded joint frames back to the arm and gripper. Defaults to the
`action` column; `--use-state` replays `observation.state` instead.

```bash
ros2 run ur5_lerobot_data_collection data_replay \
  --dataset-path $HOME/work/all_datasets/1_std_datasets/apple_to_basket_encoded \
  --episode 0 --send-mode single --controller scaled
```

`--send-mode` is `single`, `chunk` or `chunked16`; `--controller` is `scaled`
(scaled joint trajectory controller) or `passthrough` (UR passthrough
trajectory controller, interpolated on the robot).

---

## 6. Dataset preprocessing

All preprocessing scripts live in
`src/ur5_lerobot_data_collection/scripts/` (they need `ffmpeg` with NVENC,
`pandas`, `pyarrow`; run them in the `lerobot` env). Typical pipeline after
recording:

1. **Encode images to video** (NVENC) and strip image bytes from parquet.
   Output goes to `<dataset>_encoded/` next to the source:
   ```bash
   S=$HOME/work/src/ur5_lerobot_data_collection/scripts
   cd all_datasets/3_std_datasets
   python $S/encode_videos.py apple_to_basket
   python $S/encode_videos.py apple_to_basket --encoder av1_nvenc
   python $S/encode_videos.py ~/some/where/apple_to_basket        # full path also works
   ```
2. **Combine** every `*_encoded` folder in a directory into one dataset.
   Folders matching `*combined*` are skipped so a previous merge is not
   re-merged (`--exclude` to change):
   ```bash
   python $S/combine_encoded.py --dry-run
   python $S/combine_encoded.py --output 3_combined_encoded --chunks-size 1000
   python $S/combine_encoded.py --root ~/work/all_datasets/3_std_datasets --output 3_combined_encoded
   ```
3. **Optional, shift actions** so `action[t] = state[t+1]` (realised next
   state for the arm, next command for the gripper):
   ```bash
   python $S/convert_action.py <src_encoded> <dst_encoded_shifted>
   ```
4. **Optional, reorder joints** for old datasets recorded before
   `shoulder_pan` was moved to index 0:
   ```bash
   python $S/reorder_joints.py --dry-run        # every sub-directory of cwd
   python $S/reorder_joints.py <dataset>        # writes <dataset>_reordered/ next to it
   ```

Dataset generations: `1_std_datasets` (first fruit set), `2_std_datasets`,
`3_std_datasets` (729 episodes, 232k frames combined), `4_std_datasets`. Each
has nine task folders: `{apple,mango,pepper}_to_{basket,white,wooden}`.

GR00T needs a `meta/modality.json` mapping `state.ur5_arm`, `state.gripper`,
`action.ur5_arm`, `action.gripper`, `video.azure_kinect`, `video.realsense`
onto the LeRobot columns. `pkgs/Isaac-GR00T/scripts/load_dataset.py` and
`test_encoded_dataset.py` verify a dataset loads.

---

## 7. Fine-tuning

All training runs on the same LeRobot dataset folders. Checkpoints go under
`models/` (gitignored). Naming convention used so far:
`<model>_<episodes>x<action_dim>D_E<epochs>B<batch>[_R<lora_rank>]_<taskset>`.

### 7.1 GR00T N1.5

conda env `gr00t_n1.5`, run from `pkgs/Isaac-GR00T`. Data config
`ur5_2f85_arm_gripper` (defined in `gr00t/experiment/data_config.py`),
embodiment tag `new_embodiment`.

```bash
dataset_list=(
    "$HOME/work/all_datasets/3_std_datasets/apple_to_basket_encoded"
    "$HOME/work/all_datasets/3_std_datasets/apple_to_white_encoded"
    # ... one entry per task folder, or a single combined dataset
)

python3 scripts/gr00t_finetune.py \
    --dataset-path ${dataset_list[@]} \
    --output-dir $HOME/work/models/gr00t/gr00t_81x9D_E32B_R32_fruit \
    --data-config ur5_2f85_arm_gripper \
    --embodiment-tag new_embodiment \
    --max-steps 30000 \
    --save-steps 10000 \
    --batch-size 32 \
    --gradient-accumulation-steps 4 \
    --lora-rank 16 --lora-alpha 16 \
    --video-backend decord
```

Key flags (defaults in `scripts/gr00t_finetune.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--tune-llm / --tune-visual` | false | unfreeze Eagle LLM / vision tower |
| `--tune-projector / --tune-diffusion-model` | true | train projector and DiT action head |
| `--lora-rank` | 0 (off) | LoRA on the backbone; 16 or 32 used here |
| `--num-gpus` | 1 | multi-GPU via torchrun |
| `--balance-dataset-weights` | true | equalise sampling across the dataset list |
| `--report-to` | wandb | logging backend |

### 7.2 GR00T N1.5 with SmolVLA action expert

Same env and data. Loads the pretrained Eagle backbone from
`nvidia/GR00T-N1.5-3B` and attaches a freshly initialised SmolVLA-style
flow-matching expert, trained from scratch. Data config
`ur5_gr00tsmolvla`.

```bash
python3 scripts/gr00t_finetune_smolvla.py \
    --dataset-path ${dataset_list[@]} \
    --output-dir $HOME/work/models/gr00tsmolvla/gr00tsmolvla_81x9D_E64B_fruit \
    --data-config ur5_gr00tsmolvla \
    --embodiment-tag new_embodiment \
    --max-steps 30000 --save-steps 10000 \
    --batch-size 32 --gradient-accumulation-steps 4 \
    --lora-rank 16
```

Expert-specific flags: `--expert-hidden-size 720`, `--num-layers 16`,
`--num-heads 12`, `--self-attn-every-n-layers 2`, `--num-steps 10`
(flow-matching denoising steps). Design rationale is in
`pkgs/groot_smolvla_action_expert_swap.md` and
`project_notes/groot_smolvla_action_expert_impl.md`.

### 7.3 SmolVLA

conda env `lerobot`, run from `pkgs/lerobot`.

```bash
python3 src/lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=ur5_combined \
  --dataset.root=$HOME/work/all_datasets/3_std_datasets/3_combined_encoded \
  --batch_size=64 \
  --steps=100000 \
  --save_freq=10000 \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --output_dir=$HOME/work/models/smolvla/smolvla_ur5_fruit \
  --num_workers=8
```

Checkpoints land in `<output_dir>/checkpoints/<step>/pretrained_model`.

### 7.4 pi0 / pi0-FAST (openpi)

conda env `openpi`, run from `pkgs/openpi` with `uv run`. Configs
`pi0_ur5`, `pi0_ur5_lora`, `pi0_fast_ur5`, `pi0_fast_ur5_lora` are in
`src/openpi/training/config.py`; `pi0_ur5_lora` also freezes the SigLIP
vision tower and fits a single 24 GB GPU.

```bash
export LEROBOT_HOME=$HOME/work/all_datasets/3_std_datasets     # parent of the dataset folder

# once per dataset: normalisation stats
uv run scripts/compute_norm_stats.py --config-name pi0_ur5_lora

# train (repo_id defaults to 3_combined_encoded; override with --data.repo_id=<folder>)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_ur5_lora \
  --exp-name pi0_ur5_lora_D243 --overwrite
```

Convert a JAX checkpoint to PyTorch if needed:

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint-dir $HOME/work/models/pi0_ur5_lora/pi0_ur5_lora/29999 \
    --config-name pi0_ur5_lora \
    --output-path $HOME/work/models/pi0_ur5_lora/pytorch_29999 \
    --precision bfloat16
# the converter looks in the wrong place for norm stats on step-dir checkpoints:
cp -r $HOME/work/models/pi0_ur5_lora/pi0_ur5_lora/29999/assets \
      $HOME/work/models/pi0_ur5_lora/pytorch_29999/
```

---

## 8. Offline evaluation on a dataset

Plays recorded observations through the policy and plots predicted vs
ground-truth actions. `--chunk-filter` and `--boundary-blend` mirror the
smoothing options used on the robot (`project_notes/filter_utils_explained.md`).

GR00T (either head):

```bash
cd pkgs/Isaac-GR00T
python scripts/eval_policy.py --plot \
  --model-path $HOME/work/models/gr00t/gr00t_81x9D_E32B_R32_fruit \
  --dataset-path $HOME/work/all_datasets/2_std_datasets/test_encoded \
  --embodiment-tag new_embodiment \
  --data-config ur5_2f85_arm_gripper \
  --modality-keys ur5_arm gripper \
  --start_traj=1 --steps=120 \
  --video-backend decord \
  --denoising-steps=8 \
  --action_horizon=16 --chunk-filter rts --boundary-blend
```

`--chunk-filter` accepts `none`, `savgol`, `rts` (Rauch-Tung-Striebel
Kalman smoother).

SmolVLA:

```bash
cd pkgs/lerobot
python3 src/lerobot/scripts/eval_lerobot_policies.py \
    --policy_path $HOME/work/models/smolvla/smolvla_ur5_fruit/checkpoints/100000/pretrained_model \
    --dataset_root $HOME/work/all_datasets/3_std_datasets/apple_to_basket_encoded \
    --episodes 1 --steps 120 \
    --chunk_filter rts --boundary_blend
```

openpi:

```bash
cd pkgs/openpi
uv run python scripts/eval_policy.py \
    --config pi0_ur5_lora \
    --checkpoint-dir $HOME/work/models/pi0_ur5_lora_D243/59999 \
    --dataset-path $HOME/work/all_datasets/2_std_datasets/test_encoded \
    --trajs 5 --plot
```

GR00T can also replay a dataset's observations on the real robot
(`scripts/eval_policy_hardware.py`) with the driver in forward-position mode:

```bash
python scripts/eval_policy_hardware.py \
  --model-path $HOME/work/models/gr00t/gr00t_64x9D_E128B_fruit_shift \
  --dataset-path $HOME/work/all_datasets/1_std_datasets/test_fruit_encoded_shift \
  --embodiment-tag new_embodiment --data-config ur5_2f85_arm_gripper \
  --modality-keys ur5_arm gripper \
  --start_traj=1 --steps=300 --filter --send-mode single \
  --denoising-steps=8 --action_horizon=16 --dt=0.3
```

---

## 9. Deployment on the real robot

Deployment is a server plus a client. The server holds the model and answers
ZMQ requests; the client runs the sensors and the robot. They can be on
different machines (`--host`, `--port`).

Wire protocol (identical for all servers):

```
request : video.azure_kinect (1,H,W,3) u8, video.wfov (1,H,W,3) u8,
          state.ur5_arm (1,6) f64, state.gripper (1,1) f64,
          annotation.human.task_description [str]
response: action.ur5_arm (H,6) f64, action.gripper (H,) f64
```

### 9.1 Start an inference server

GR00T N1.5 (env `gr00t_n1.5`, from `pkgs/Isaac-GR00T`):

```bash
python scripts/inference_service.py --server \
  --model-path $HOME/work/models/gr00t/gr00t_81x9D_E128B_fruit \
  --data-config ur5_2f85_arm_gripper \
  --embodiment-tag new_embodiment --port 5555
```

GR00T + SmolVLA expert (same script, different data config):

```bash
python scripts/inference_service.py --server \
  --model-path $HOME/work/models/gr00tsmolvla/gr00tsmolvla_81x9D_E64B_fruit \
  --data-config ur5_gr00tsmolvla \
  --embodiment-tag new_embodiment --port 5555
```

SmolVLA (env `lerobot`, from `pkgs/lerobot`):

```bash
python3 src/lerobot/scripts/server/smolvla_inference_service.py --server \
  --model-path $HOME/work/models/smolvla/smolvla_ur5_fruit/checkpoints/100000/pretrained_model \
  --port 5555
```

openpi (env `openpi`, from `pkgs/openpi`):

```bash
uv run python scripts/openpi_inference_service.py --server \
  --config pi0_ur5_lora \
  --checkpoint-dir $HOME/work/models/pi0_ur5_lora_D243/59999 \
  --port 5555
```

Smoke test any server with synthetic observations, no robot needed:

```bash
python pkgs/openpi/scripts/openpi_inference_service.py --client --port 5555
python -m lerobot.scripts.server.smolvla_inference_service --client --port 5555
```

### 9.2 Bring up the robot

1. UR driver in the mode matching the client (section 4.1).
2. Gripper (section 4.2).
3. Put the arm near the home joint pose. The clients home the arm through
   the forward-position controller before starting.

### 9.3 Run a client

All clients live in `pkgs/Isaac-GR00T/scripts` and run in env `gr00t_n1.5`
with `install/setup.bash` sourced. They share `ur5_client_common.py`
(sensor node, camera capture, homing, gripper pacing) and `filter_utils.py`.
Full design of the async client and the controller trade-offs is in
`scripts/gr00t_client_guide.md` and
`project_notes/ur5_streaming_controller_analysis.md`.

| Client | Driver controller | Behaviour |
|---|---|---|
| `ur5_gr00t_streaming_client.py` (production) | scaled JTC (default) or forward_position | Two threads: inference thread blends new chunks into a pending queue, control thread pops one action per `dt`. No idle time between chunks. |
| `ur5_gr00t_timed_chunk_client.py` | scaled JTC | Timed chunk sending with configurable aggregation. |
| `ur5_gr00t_simple_client.py` | scaled JTC or forward_position | Synchronous: infer, execute the whole chunk, repeat. Robot idles during inference. |

**Streaming client** (recommended). Defaults are already the production
settings: `jtc_topic` interface, `ramp` aggregation, chunk threshold 0.2,
lookahead 30, continue end-velocity, Hermite segments plus 8 Hz Butterworth
on the 125 Hz stream.

```bash
python scripts/ur5_gr00t_streaming_client.py \
  --lang "place apple in the basket." \
  --action_horizon 50 \
  --dt 0.05
```

Useful options:

| Flag | Default | Note |
|---|---|---|
| `--host / --port` | localhost / 5555 | remote server |
| `--control_interface` | jtc_topic | `forward_position` for the dense Python stream |
| `--aggregate_fn_name` | ramp | `weighted_average`, `latest_only`, `average`, `conservative` |
| `--segment-interp` | hermite | `linear` reproduces the old behaviour |
| `--stream-filter` | butter | `none`, `oneeuro`; `--stream-filter-cutoff 8` |
| `--use-speed-scaling` | true | honour the pendant speed slider |
| `--log` | off | write `streaming_log.npz` |

Analyse a run:

```bash
python joint_log/plot_streaming_log.py streaming_log.npz   # jerk / acc / dv stats, *_stream.png
python scripts/compare_logs.py simple_client_log.npz streaming_log.npz   # A/B simple vs streaming
```

**Simple client**:

```bash
# scaled JTC, most smoothing
python scripts/ur5_gr00t_simple_client.py --send-mode chunk \
  --lang "place apple in the basket." --chunk-filter rts --boundary-blend \
  --action_horizon 30 --dt 0.07

# forward position controller
python scripts/ur5_gr00t_simple_client.py --controller forward_position_controller --send-mode chunk
```

**Timed chunk client**:

```bash
python scripts/ur5_gr00t_timed_chunk_client.py \
  --lang "place apple in the basket." --host <server-ip> --port 5555 \
  --chunk-filter rts --aggregate_fn_name ramp --action_horizon 30 --dt 0.07
```

Per-model notes:

- SmolVLA and pi0 return longer chunks. Use `--action_horizon 50` (SmolVLA)
  or `--action_horizon 30 --dt 0.07` (pi0) with the same clients.
- The GR00T action horizon is 16 by default; `--action_horizon` on the
  client caps how many steps of a chunk are executed.

---

## 10. Analysis and visualisation tools

| Script | Purpose |
|---|---|
| `pkgs/Isaac-GR00T/scripts/visualize_text_image_cross_attention.py` | Eagle text-to-image cross-attention maps; optional Grounding DINO boxes (`--box_source gdino`) |
| `pkgs/Isaac-GR00T/scripts/visualize_eagle_attention.py`, `attention_bbox.py` | attention scoring inside bounding boxes |
| `all_datasets/plot_joint_observations.py` | compare joint observations across datasets |
| `project_notes/*.md` | jerk metrics, inter-chunk plots, filter maths, streaming latency study |

Example:

```bash
python scripts/visualize_text_image_cross_attention.py \
  --image_path scripts/images/fruit1.png --text_query mango \
  --box_source gdino --box_model IDEA-Research/grounding-dino-base \
  --box_negatives 'lemon,banana,apple,green pepper,yellow pepper' --box_threshold 0.15
```

---

## 11. Quick reference: full deployment checklist

1. `./ubuntu22.04_ros2/run_lerobot_env.sh` and `source install/setup.bash`.
2. Terminal A: UR driver with `scaled_joint_trajectory_controller` (4.1).
3. Terminal B: gripper (4.2).
4. Terminal C: inference server for the chosen model (9.1).
5. Terminal D: `conda activate gr00t_n1.5`, then the streaming client (9.3).
6. Press the pendant play button so External Control connects, watch the
   arm home, then the task runs for `--num_steps` ticks.
