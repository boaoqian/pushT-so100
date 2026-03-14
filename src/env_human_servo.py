import os
# 离屏渲染配置
os.environ["MUJOCO_GL"] = "egl"

import time
import threading
import logging
import mujoco
import mujoco.viewer
import numpy as np
import pygame
import cv2
import torch
from scipy.spatial.transform import Rotation as R
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from helper import *

# --- 配置 logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 导入 LeRobot 相关库

# --- 全局配置参数 ---
MOVE_SPEED = 0.05
ROT_SPEED = 1.0
DEADZONE = 0.1
FPS = 10
VIDEO_STEP = 1.0 / FPS
OBS_IMAGES_SHAPE = (224, 224, 3)

#环境配置
pos_random_range = 0.05

# LeRobot 数据集路径与参数
REPO_ID = "/media/qba/Data/Project/Robot/So100PushT/data/NewData3.7-delta"
ROBOT_TYPE = "so100_arm"
USE_DELTA_ACTION = True
TOLERANCE = 0.001  # 动作去重容差

# 录制状态
is_recording = False
record_buffer = []  # 暂存当前 Episode 的帧数据
save_thread = None # 用于追踪后台线程

# 防止按键抖动
buttonCooldown = 0.0
COOLDOWN_SEC = 0.5

# --- 加载 MuJoCo 模型 ---
XML_PATH = "./chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/human_env.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# 关节与 Mocap ID 获取
act_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in act_names]
mocap_name = "target_mocap"
mocap_id = model.body(mocap_name).mocapid[0]

# 目标物体 T_block 的地址
t_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "T_block")
t_qpos_adr = model.jnt_qposadr[model.body_jntadr[t_body_id]]
is_success = False

dataset = None
# --- 初始化 LeRobot 数据集函数 ---
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
            "dtype": "float32", "shape": (5,), "names": act_names
        }
    }

    if os.path.exists(repo_path):
        logger.info(f"检测到现有数据集，正在加载: {repo_path}")
        return LeRobotDataset(repo_path)
    else:
        logger.info(f"未检测到数据集，正在创建新仓库: {repo_path}")
        return LeRobotDataset.create(
            repo_id=repo_path,
            fps=FPS,
            robot_type=ROBOT_TYPE,
            features=features
        )

def async_save_to_lerobot(frames_list):
    """ 在独立线程中处理数据并写入磁盘 """
    global dataset
    if not dataset:
        dataset = init_lerobot_dataset(repo_path=REPO_ID)
    if len(frames_list) < 2:
        return
        
    num_frames = len(frames_list) - 1
    logger.info(f"[后台线程] 开始处理新 Episode ({num_frames} 帧)，模式: {'Delta' if USE_DELTA_ACTION else 'Absolute'}")
    
    last_pushed_action = None
    added_count = 0
    
    for i in range(num_frames):
        curr_obs_frame = frames_list[i]
        next_obs_frame = frames_list[i+1]
        
        # --- 计算 Action ---
        if USE_DELTA_ACTION:
            # Delta = 下一时刻的状态 - 当前时刻的状态
            target_action = (next_obs_frame["state"] - curr_obs_frame["state"]).astype(np.float32)
        else:
            # 绝对位置 = 下一时刻的状态
            target_action = next_obs_frame["action"].astype(np.float32)
        
        # --- 动作去重逻辑 ---
        # 如果是 Delta 模式，且增量几乎为 0，说明机械臂没动
        if USE_DELTA_ACTION:
            if np_allabs(target_action):
                continue
        else:
            if last_pushed_action is not None and np.allclose(target_action, last_pushed_action, atol=TOLERANCE):
                continue
        
        # 构造数据帧
        frame_data = {
            "observation.images.cam_top": curr_obs_frame["cam_top"],
            "observation.images.cam_side": curr_obs_frame["cam_side"],
            "observation.state": curr_obs_frame["state"].astype(np.float32),
            "action": target_action,
            "task": "pushT"
        }
        
        dataset.add_frame(frame_data)
        last_pushed_action = target_action
        added_count += 1
    
    if added_count > 0:
        dataset.save_episode()
        logger.info(f"[后台线程] 存储完成！模式: {'Delta' if USE_DELTA_ACTION else 'Absolute'}, 有效帧: {added_count} 总数: {dataset.num_episodes}")
    else:
        logger.warning("[后台线程] 跳过：无有效动作变化。")

# 辅助函数：判断绝对值是否都小于容差
def np_allabs(x):
    return np.all(np.abs(x) < TOLERANCE)

# --- 手柄与录制控制 ---
def record_toggle():
    global is_recording, record_buffer, save_thread
    if is_recording:
        is_recording = False
        logger.info("录制停止。")
        if len(record_buffer) > 0:
            # 开启新线程处理保存，避免阻塞主循环
            save_thread = threading.Thread(
                target=async_save_to_lerobot, 
                args=(list(record_buffer),), # 传入 buffer 的副本
                daemon=True
            )
            save_thread.start()
        record_buffer = []
    else:
        logger.info("录制开始 (Recording ON)")
        record_buffer = []
        is_recording = True

def reset_env():
    global is_success, is_recording, record_buffer, pos_random_range
    is_success = False
    is_recording = False
    record_buffer.clear()
    key_id = 0
    # 重置模型
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.mocap_pos[:] = model.key_mpos[key_id]
    data.mocap_quat[:] = model.key_mquat[key_id]
    # 随机化物体位置
    data.qpos[t_qpos_adr:t_qpos_adr+2] = [np.random.uniform(0.3-pos_random_range, 0.3+pos_random_range), np.random.uniform(-pos_random_range, pos_random_range)]
    data.qpos[t_qpos_adr+2] = 0.01
    rad = np.random.uniform(-0.5,0.5)
    data.qpos[t_qpos_adr+3:t_qpos_adr+7] = [np.cos(rad), 0.0, 0.0, np.sin(rad)]
    mujoco.mj_forward(model, data)


# --- 初始化硬件输入 ---
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
if joystick: joystick.init()
else:
    logger.error("未检测到手柄")
    exit()

def joystick_control():
    global buttonCooldown
    if not joystick: return False
    pygame.event.pump()

    # 位置控制 (左摇杆)
    ax0, ax1 = joystick.get_axis(0), joystick.get_axis(1)
    dx = (abs(ax0) > DEADZONE) * ax0 * MOVE_SPEED * model.opt.timestep
    dy = -(abs(ax1) > DEADZONE) * ax1 * MOVE_SPEED * model.opt.timestep
    dz = (joystick.get_button(4) - joystick.get_button(0)) * MOVE_SPEED * model.opt.timestep
    data.mocap_pos[mocap_id] += np.array([dx, dy, dz])

    # 旋转控制 (右摇杆)
    ax2 = joystick.get_axis(2)
    dr = -(abs(ax2) > DEADZONE) * ax2 * ROT_SPEED * model.opt.timestep
    if abs(dr) > 0:
        q = data.mocap_quat[mocap_id]
        r_curr = R.from_quat([q[1], q[2], q[3], q[0]])
        new_q = (R.from_euler('y', dr) * r_curr).as_quat()
        data.mocap_quat[mocap_id] = [new_q[3], new_q[0], new_q[1], new_q[2]]

    now = time.time()
    # X 键 (3): 重置环境
    if joystick.get_button(3) and (now - buttonCooldown > COOLDOWN_SEC):
        buttonCooldown = now
        reset_env()
    # B 键 (1): 录制开关
    if joystick.get_button(1) and (now - buttonCooldown > COOLDOWN_SEC):
        buttonCooldown = now
        record_toggle()
    
    return joystick.get_button(11) # Start 键退出

# --- 主循环 ---
renderer = mujoco.Renderer(model, height=OBS_IMAGES_SHAPE[0], width=OBS_IMAGES_SHAPE[1])
reset_env()
video_time = 0.0

logger.info("控制就绪: X重置, B录制开关, Start退出")

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # IK 模拟与步进
            data.ctrl[act_ids] = data.qpos[act_ids] 
            if joystick_control(): break
            mujoco.mj_step(model, data)

            # 检查成功状态
            ok, dxy, dyaw = check_xy_pose_match(model, data, "T_sign_anchor", "T_block_anchor")
            if ok and not is_success:
                logger.info(f"任务成功! 误差: {dxy:.4f}")
                is_success = True

            # 视觉渲染与数据采样
            video_time += model.opt.timestep
            if video_time >= VIDEO_STEP:
                video_time = 0.0
                
                renderer.update_scene(data, camera="top_view")
                img_top = renderer.render().copy()
                renderer.update_scene(data, camera="side_view")
                img_side = renderer.render().copy()

                if is_recording:
                    record_buffer.append({
                        "cam_top": img_top,
                        "cam_side": img_side,
                        "action": data.ctrl[act_ids].copy(),
                        "state": data.qpos[act_ids].copy()
                    })

                # OpenCV 预览
                disp_img_top = cv2.cvtColor(img_top, cv2.COLOR_RGB2BGR)
                disp_img_top = cv2.resize(disp_img_top, (OBS_IMAGES_SHAPE[1]*2, OBS_IMAGES_SHAPE[0]*2))
                disp_img_side = cv2.cvtColor(img_side, cv2.COLOR_RGB2BGR)
                disp_img_side = cv2.resize(disp_img_side, (OBS_IMAGES_SHAPE[1]*2, OBS_IMAGES_SHAPE[0]*2))
                if is_recording:
                    cv2.circle(disp_img_top, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(disp_img_top, f"REC: {len(record_buffer)}", (50, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow("MuJoCo So100 - Top View", disp_img_top)
                cv2.imshow("MuJoCo So100 - Side View", disp_img_side)
                viewer.sync()
                cv2.waitKey(1)

            # 维持物理频率
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)
finally:
    # 核心修复步骤：程序关闭时的清理逻辑
    logger.info("正在关闭程序，请稍候...")
    
    # 1. 如果还在录制，强制停止并启动保存
    if is_recording:
        record_toggle()
    
    # 2. 等待后台保存线程结束
    if save_thread and save_thread.is_alive():
        logger.info("正在等待最后一组数据保存到磁盘...")
        save_thread.join()
    
    # 3. 显式销毁数据集对象
    # 这会触发 LeRobotDataset 内部的 __del__，此时 pyarrow 等模块还在内存中
    logger.info("正在保存数据集索引并释放资源...")
    del dataset
    
    cv2.destroyAllWindows()
    pygame.quit()
    logger.info("安全退出。")