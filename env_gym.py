import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import cv2
from helper import *

XML_PATH = "./chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/human_env.xml"

class PushT(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self,max_steps=2000, render_mode=None):
        super(PushT, self).__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Box(low=-3.14159, high=3.14159, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "camera_front": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            "camera_top":   spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            "robot_joints": spaces.Box(low=-3.14159, high=3.14159, shape=(5,), dtype=np.float32),
        })
        self.current_step = 0
        self.max_steps = max_steps

        self.last_dxy = 10000
        self.last_dyaw = 360

        #加载模型
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    
        act_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
        self.act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in act_names]

        # 找到T_block
        t_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "T_block")
        t_jnt_adr = self.model.body_jntadr[t_body_id]
        self.t_qpos_adr = self.model.jnt_qposadr[t_jnt_adr]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        self.last_dxy = 10000
        self.last_dyaw = 360
        self.current_step = 0

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.t_qpos_adr:self.t_qpos_adr+3] = np.array([
        np.random.uniform(0.2, 0.4),
        np.random.uniform(-0.2, 0.2),
        0.01
    ])
        rad = np.random.uniform(0.0, 2*np.pi)
        # free joint quat: (w, x, y, z)
        self.data.qpos[self.t_qpos_adr+3:self.t_qpos_adr+7] = np.array([np.cos(rad), 0.0, 0.0, np.sin(rad)])
        mujoco.mj_forward(self.model, self.data)
        obs = self.get_observation()
        return obs, {}

    def get_observation(self):
        self.renderer.update_scene(self.data, camera="top_view")
        img_top = self.renderer.render()  # RGB uint8

        self.renderer.update_scene(self.data, camera="side_view")
        img_side = self.renderer.render()
        obs = {
            "camera_top": img_top,
            "camera_side": img_side,
            "robot_joints": self.data.qpos[self.act_ids],
        }
        return obs


    def step(self, action):
        self.data.ctrl[self.act_ids] = action
        mujoco.mj_step(self.model, self.data)
        obs = self.get_observation()
        ok, dxy, dyaw = check_xy_pose_match(self.model, self.data, "T_sign_anchor", "T_block_anchor",
                                   pos_tol=0.01, yaw_tol_deg=3.0)
        reward = 0.0
        if dxy < self.last_dxy-0.1:
            reward += 0.1
            self.last_dxy = dxy
        if dyaw < self.last_dyaw-0.1:
            reward += 0.1
            self.last_dyaw = dyaw
        if ok:
            reward += 10.0
        reward -= 0.01

        self.current_step += 1
        return obs, reward, ok, self.current_step >= self.max_steps, {}
    
    def render(self):
        obs = self.get_observation()
        top_img = cv2.cvtColor(obs["camera_top"], cv2.COLOR_RGB2BGR)
        cv2.imshow("top_view", top_img)
        side_img = cv2.cvtColor(obs["camera_side"], cv2.COLOR_RGB2BGR)
        cv2.imshow("side_view", side_img)
        cv2.waitKey(1)
    