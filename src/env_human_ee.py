import os
import argparse

# Offscreen rendering configuration
os.environ["MUJOCO_GL"] = "egl"

import time
import threading
import logging
import mujoco
import mujoco.viewer
import numpy as np
import pygame
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from helper import *

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="MuJoCo SO100 data collection script")

parser.add_argument(
    "--repo_id",
    type=str,
    default="./data/NewData3.9-ee-2d-pos",
    help="Path to the LeRobot dataset repository"
)
parser.add_argument(
    "--fps",
    type=int,
    default=10,
    help="Recording frame rate"
)
parser.add_argument(
    "--move_speed",
    type=float,
    default=0.05,
    help="Mocap translation speed"
)
parser.add_argument(
    "--rot_speed",
    type=float,
    default=1.0,
    help="Mocap rotation speed"
)

args = parser.parse_args()

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Import LeRobot related libraries
# --- Global configuration ---
MOVE_SPEED = args.move_speed
ROT_SPEED = args.rot_speed
DEADZONE = 0.1
FPS = args.fps
VIDEO_STEP = 1.0 / FPS
OBS_IMAGES_SHAPE = (224, 224, 3)

# Environment configuration
pos_random_range = 0.05

# LeRobot dataset path and parameters
REPO_ID =  Path(args.repo_id).absolute()
ROBOT_TYPE = "so100_arm"
TOLERANCE = 0.0001  # Tolerance for action deduplication

# Recording state
is_recording = False
record_buffer = []  # Temporary buffer for the current episode
save_thread = None  # Used to track the background saving thread

# Button debounce
buttonCooldown = 0.0
COOLDOWN_SEC = 0.3

# --- Load MuJoCo model ---
XML_PATH = "./chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/human_env.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Get joint and mocap IDs
act_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in act_names]
mocap_name = "target_mocap"
mocap_id = model.body(mocap_name).mocapid[0]

# Address of target object T_block
t_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "T_block")
t_qpos_adr = model.jnt_qposadr[model.body_jntadr[t_body_id]]
is_success = False

dataset = None


# --- Initialize hardware input ---
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
if joystick:
    joystick.init()
else:
    logger.error("No joystick detected")
    exit()

# --- Initialize LeRobot dataset ---
def init_lerobot_dataset(repo_path):
    features = {
        "observation.images.cam_top": {
            "dtype": "video", "shape": OBS_IMAGES_SHAPE, "names": ["channels", "height", "width"]
        },
        "observation.images.cam_side": {
            "dtype": "video", "shape": OBS_IMAGES_SHAPE, "names": ["channels", "height", "width"]
        },
        "observation.state": {
            "dtype": "float32", "shape": (5,), "names": act_names
        },
        "action": {
            "dtype": "float32", "shape": (2,), "names": ["mocap_x", "mocap_y"]
        }
    }

    if os.path.exists(repo_path):
        logger.info(f"Existing dataset detected, loading: {repo_path}")
        return LeRobotDataset(repo_path)
    else:
        logger.info(f"Dataset not found, creating new repository: {repo_path}")
        return LeRobotDataset.create(
            repo_id=repo_path,
            fps=FPS,
            robot_type=ROBOT_TYPE,
            features=features
        )

def get_mocap_4d_pose(mocap_pos, mocap_quat):
    """Extract x, y, z and Y-axis rotation from mocap data."""
    x, y, z = mocap_pos
    return np.array([x, y], dtype=np.float32)

def async_save_to_lerobot(frames_list):
    """Process data and write it to disk in a separate thread."""
    global dataset
    if not dataset:
        dataset = init_lerobot_dataset(repo_path=REPO_ID)
    if len(frames_list) < 2:
        return
        
    num_frames = len(frames_list) - 1
    logger.info(f"[Background thread] Start processing new episode ({num_frames} frames)")
    added_count = 0
    last_action = None
    
    for i in range(num_frames):
        curr_frame = frames_list[i]
        target_action = curr_frame["mocap_pose_2d"]

        # --- Action deduplication ---
        if last_action is not None and np_allabs(target_action - last_action) < TOLERANCE:
            continue
        frame_data = {
            "observation.images.cam_top": curr_frame["cam_top"],
            "observation.images.cam_side": curr_frame["cam_side"],
            "observation.state": curr_frame["state"].astype(np.float32),
            "action": target_action,
            "task": "pushT"
        }
        dataset.add_frame(frame_data)
        added_count += 1
        last_action = target_action
    
    if added_count > 0:
        dataset.save_episode()
        logger.info(f"[Background thread] Save completed, valid frames: {added_count}, total episodes: {dataset.num_episodes}")
    else:
        logger.warning("[Background thread] Skipped: no valid action changes.")

# Utility function: check whether all absolute values are below tolerance
def np_allabs(x):
    return np.all(np.abs(x) < TOLERANCE)

# --- Joystick and recording control ---
def record_toggle():
    global is_recording, record_buffer, save_thread
    if is_recording:
        is_recording = False
        logger.info("Recording stopped.")
        if len(record_buffer) > 0:
            # Start a new thread for saving to avoid blocking the main loop
            save_thread = threading.Thread(
                target=async_save_to_lerobot,
                args=(list(record_buffer),),
                daemon=True
            )
            save_thread.start()
        record_buffer = []
    else:
        logger.info("Recording started (Recording ON)")
        record_buffer = []
        is_recording = True

def reset_env():
    global is_success, is_recording, record_buffer, pos_random_range
    is_success = False
    is_recording = False
    record_buffer.clear()
    key_id = 0

    # Reset model state
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.mocap_pos[:] = model.key_mpos[key_id]
    data.mocap_quat[:] = model.key_mquat[key_id]

    # Randomize object position
    data.qpos[t_qpos_adr:t_qpos_adr+2] = [
        np.random.uniform(0.25 - pos_random_range, 0.25 + pos_random_range),
        np.random.uniform(-pos_random_range, pos_random_range)
    ]
    data.qpos[t_qpos_adr+2] = 0.01
    rad = np.random.uniform(-3.14, 3.14)
    data.qpos[t_qpos_adr+3:t_qpos_adr+7] = [np.cos(rad), 0.0, 0.0, np.sin(rad)]
    mujoco.mj_forward(model, data)

def joystick_control():
    global buttonCooldown
    if not joystick:
        return False
    pygame.event.pump()

    # Position control (left stick)
    ax0, ax1 = joystick.get_axis(0), joystick.get_axis(1)
    dx = (abs(ax0) > DEADZONE) * ax0 * MOVE_SPEED * model.opt.timestep
    dy = -(abs(ax1) > DEADZONE) * ax1 * MOVE_SPEED * model.opt.timestep
    dz = (joystick.get_button(4) - joystick.get_button(0)) * MOVE_SPEED * model.opt.timestep
    data.mocap_pos[mocap_id] += np.array([dx, dy, dz])

    # Rotation control (right stick)
    ax2 = joystick.get_axis(2)
    dr = -(abs(ax2) > DEADZONE) * ax2 * ROT_SPEED * model.opt.timestep
    if abs(dr) > 0:
        q = data.mocap_quat[mocap_id]
        r_curr = R.from_quat([q[1], q[2], q[3], q[0]])
        new_q = (R.from_euler('y', dr) * r_curr).as_quat()
        data.mocap_quat[mocap_id] = [new_q[3], new_q[0], new_q[1], new_q[2]]

    now = time.time()

    # X button (3): reset environment
    if joystick.get_button(3) and (now - buttonCooldown > COOLDOWN_SEC):
        buttonCooldown = now
        reset_env()

    # B button (1): toggle recording
    if joystick.get_button(1) and (now - buttonCooldown > COOLDOWN_SEC):
        buttonCooldown = now
        record_toggle()
    
    return joystick.get_button(11)  # Start button exits

# --- Main loop ---
renderer = mujoco.Renderer(model, height=OBS_IMAGES_SHAPE[0], width=OBS_IMAGES_SHAPE[1])
reset_env()
video_time = 0.0

logger.info("Control ready: X=reset, B=toggle recording, Start=exit")

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # IK simulation and physics stepping
            data.ctrl[act_ids] = data.qpos[act_ids]
            if joystick_control():
                break
            mujoco.mj_step(model, data)

            # Check task success
            ok, dxy, dyaw = check_xy_pose_match(model, data, "T_sign_anchor", "T_block_anchor")
            if ok and not is_success:
                logger.info(f"Task succeeded! Position error: {dxy:.4f}")
                is_success = True

            # Visual rendering and data sampling
            video_time += model.opt.timestep
            if video_time >= VIDEO_STEP:
                video_time = 0.0
                
                renderer.update_scene(data, camera="top_view")
                img_top = renderer.render().copy()
                renderer.update_scene(data, camera="side_view")
                img_side = renderer.render().copy()

                # OpenCV preview
                disp_img_top = cv2.cvtColor(img_top, cv2.COLOR_RGB2BGR).copy()
                disp_img_top = cv2.resize(disp_img_top, (OBS_IMAGES_SHAPE[1] * 2, OBS_IMAGES_SHAPE[0] * 2))
                disp_img_side = cv2.cvtColor(img_side, cv2.COLOR_RGB2BGR).copy()
                disp_img_side = cv2.resize(disp_img_side, (OBS_IMAGES_SHAPE[1] * 2, OBS_IMAGES_SHAPE[0] * 2))

                # Get current mocap 2D pose
                current_mocap_2d = get_mocap_4d_pose(
                    data.mocap_pos[mocap_id].copy(),
                    data.mocap_quat[mocap_id].copy()
                )

                if is_recording:
                    record_buffer.append({
                        "cam_top": img_top,
                        "cam_side": img_side,
                        "mocap_pose_2d": current_mocap_2d,
                        "state": data.qpos[act_ids].copy()
                    })

                    cv2.circle(disp_img_top, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(
                        disp_img_top, f"REC: {len(record_buffer)}", (50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                    )
                
                info_texts = [
                    f"Mocap X: {current_mocap_2d[0]:.3f}",
                    f"Mocap Y: {current_mocap_2d[1]:.3f}",
                ]

                for i, text in enumerate(info_texts):
                    cv2.putText(
                        disp_img_top, text, (20, 80 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )

                cv2.imshow("MuJoCo So100 - Top View", disp_img_top)
                cv2.imshow("MuJoCo So100 - Side View", disp_img_side)
                viewer.sync()
                cv2.waitKey(1)

            # Maintain physics rate
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

finally:
    # Cleanup logic when the program exits
    logger.info("Shutting down safely...")

    if save_thread and save_thread.is_alive():
        logger.info("Waiting for the last batch of data to be saved...")
        save_thread.join()

    logger.info("Saving dataset index and releasing resources...")
    del dataset

    cv2.destroyAllWindows()
    pygame.quit()
    logger.info("Exit complete.")