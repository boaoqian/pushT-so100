import os
os.environ["MUJOCO_GL"] = "egl"   # 离屏渲染，避免和窗口系统/SDL冲突

import time
import pickle
import mujoco
import mujoco.viewer
import numpy as np
import pygame
import cv2
from scipy.spatial.transform import Rotation as R  # 处理旋转
from helper import *

#配置参数
MOVE_SPEED = 0.1
ROT_SPEED = 1.0
DEADZONE = 0.1

FPS = 30
VIDEO_STEP = 1.0 / FPS

#录制状态
is_recording = False
record_buffer = {}  # 存储每帧数据

if not os.path.exists("RecordTemp"):
    os.mkdir("RecordTemp")

#防止按住按钮一直触发
buttonCooldown = 0.0
COOLDOWN_SEC = 0.5

#加载模型
XML_PATH = "./chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/world.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# 关节与 Mocap 配置
act_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in act_names]

MOCAP_NAME = "target_mocap"
mocap_id = model.body(MOCAP_NAME).mocapid[0]

# 找到T_block
t_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "T_block")
t_jnt_adr = model.body_jntadr[t_body_id]
t_qpos_adr = model.jnt_qposadr[t_jnt_adr]
is_success = False

# --- 初始化 Pygame 手柄 ---
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
if joystick:
    joystick.init()
    print(f"成功连接手柄: {joystick.get_name()}")
else:
    print("未找到手柄，请检查连接。")
    exit()

def random_t_pos():
    global is_success
    is_success = False
    data.qpos[t_qpos_adr:t_qpos_adr+3] = np.array([
        np.random.uniform(0.2, 0.4),
        np.random.uniform(-0.2, 0.2),
        0.01
    ])
    rad = np.random.uniform(0.0, 2*np.pi)
    # free joint quat: (w, x, y, z)
    data.qpos[t_qpos_adr+3:t_qpos_adr+7] = np.array([np.cos(rad), 0.0, 0.0, np.sin(rad)])
    mujoco.mj_forward(model, data)

def record_toggle():
    """切换录制：停止时保存 pkl"""
    global is_recording, record_buffer
    is_recording = not is_recording
    if is_recording:
        print("录制状态: ON")
        record_buffer["cam_top"] = []
        record_buffer["cam_side"] = []
        record_buffer["action"] = []
        record_buffer["pose"] = []
    else:
        if len(record_buffer["cam_top"]) > 0:
            filename = f"RecordTemp/record-{int(time.time())}.pkl"
            record_buffer["action"] = record_buffer["action"][1:]
            record_buffer["action"].append(record_buffer["action"][-1])
            with open(filename, "wb") as f:
                pickle.dump(record_buffer, f)
            print(f"录制状态: OFF，已保存: {filename}，帧数: {len(record_buffer['cam_top'])}")
            record_buffer.clear()
        else:
            print("录制状态: OFF（无数据）")
        record_buffer.clear()

def joystick_control():
    """手柄控制 + 状态按钮（重置/录制/退出）"""
    global buttonCooldown
    if not joystick:
        return False  # 不退出

    pygame.event.pump()

    # 位置控制 (左摇杆 + A/Y键)
    ax0 = joystick.get_axis(0)
    ax1 = joystick.get_axis(1)
    ax2 = joystick.get_axis(2)

    dx = (abs(ax0) > DEADZONE) * ax0 * MOVE_SPEED * model.opt.timestep
    dy = -(abs(ax1) > DEADZONE) * ax1 * MOVE_SPEED * model.opt.timestep
    dz = (joystick.get_button(4) - joystick.get_button(0)) * MOVE_SPEED * model.opt.timestep  # Y - A

    data.mocap_pos[mocap_id] += np.array([dx, dy, dz])

    # 旋转控制 (右摇杆 axis2) 绕 y 轴
    dr = -(abs(ax2) > DEADZONE) * ax2 * ROT_SPEED * model.opt.timestep
    if abs(dr) > 0:
        q = data.mocap_quat[mocap_id]  # (w,x,y,z)
        r_curr = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy: (x,y,z,w)
        new_q = (R.from_euler('y', dr) * r_curr).as_quat()  # (x,y,z,w)
        data.mocap_quat[mocap_id] = [new_q[3], new_q[0], new_q[1], new_q[2]]  # back to (w,x,y,z)

    now = time.time()

    # X键 (3) -> 重置
    if joystick.get_button(3) and (now - buttonCooldown > COOLDOWN_SEC):
        buttonCooldown = now
        mujoco.mj_resetData(model, data)
        random_t_pos()
        print("仿真已重置")

    # B键 (1) -> 录制开关
    if joystick.get_button(1) and (now - buttonCooldown > COOLDOWN_SEC):
        buttonCooldown = now
        record_toggle()

    # Start (11) -> 退出
    if joystick.get_button(11):
        return True

    return False

def solve_ik_stub():
    data.ctrl[act_ids] = data.qpos[act_ids]

#渲染器
renderer = mujoco.Renderer(model, height=480, width=640)

print("控制说明: X重置, B录制开关, Start退出;")
random_t_pos()

video_time = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        solve_ik_stub()
        want_quit = joystick_control()
        if want_quit:
            break

        mujoco.mj_step(model, data)
        ok, dxy, dyaw = check_xy_pose_match(model, data, "T_block", "T_sign",
                                   pos_tol=0.03, yaw_tol_deg=5.0)
        if ok and not is_success:
            print("成功!")
            is_success = True

        # 帧率控制
        video_time += model.opt.timestep
        if video_time >= VIDEO_STEP:
            video_time = 0.0

            renderer.update_scene(data, camera="top_view")
            img_top = renderer.render()  # RGB uint8

            renderer.update_scene(data, camera="side_view")
            img_side = renderer.render()

            combined = np.hstack([img_top, img_side])  # (H, W*2, 3)
            display_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

            #录制指示
            if is_recording:
                cv2.circle(display_bgr, (30, 30), 10, (0, 0, 255), -1)

            cv2.imshow("Robot Observation (Top | Side)", display_bgr)

            #录制逻辑
            if is_recording:
                    record_buffer["cam_top"].append(img_top)
                    record_buffer["cam_side"].append(img_side)
                    record_buffer["action"].append(data.ctrl[act_ids])
                    record_buffer["pose"].append(data.qpos[act_ids])

            # viewer 同步+cv2更新
            viewer.sync()
            cv2.waitKey(1)

        # 物理模拟帧率控制
        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)

#清理
cv2.destroyAllWindows()
pygame.quit()
print("程序退出")