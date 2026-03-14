import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import cv2
from helper import *

class PushT(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, xml_path, max_steps=1000, render_mode=None):
        super(PushT, self).__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.pos_random_range = 0.05
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=224, width=224)

        self.action_space = spaces.Box(
            low=np.array([-1, -1]), 
            high=np.array([1, 1]), 
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "cam_top": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            "cam_side": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            "observation.state": spaces.Box(low=-10.0, high=10.0, shape=(5,), dtype=np.float32),
        })

        self.act_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
        self.act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.act_names]
        self.mocap_id = self.model.body("target_mocap").mocapid[0]
        
        t_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "T_block")
        self.t_qpos_adr = self.model.jnt_qposadr[self.model.body_jntadr[t_body_id]]

        self.current_step = 0
        self.key_id = 0
        self._obs = None

    def _set_mocap_2d(self, action):
            """ 
            action: [x,y]
            """
            dx, dy = action
            org_pos = self.data.mocap_pos[self.mocap_id]
            self.data.mocap_pos[self.mocap_id] = [dx, dy, org_pos[2]]
    def get_observation(self):
        self.renderer.update_scene(self.data, camera="top_view")
        img_top = self.renderer.render().copy()

        self.renderer.update_scene(self.data, camera="side_view")
        img_side = self.renderer.render().copy()

        obs = {
            "cam_top": img_top,
            "cam_side": img_side
        }
        obs |= {k:v for k,v in zip(self.act_names ,self.data.qpos[self.act_ids].copy())}
        self._obs = obs
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: np.random.seed(seed)
        
        self.current_step = 0
        # reset to keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        
        # ramdomize T_block pos
        self.data.qpos[self.t_qpos_adr : self.t_qpos_adr+2] = [
            np.random.uniform(0.25 - self.pos_random_range, 0.25+self.pos_random_range),
            np.random.uniform(-self.pos_random_range, self.pos_random_range)
        ]
        self.data.qpos[self.t_qpos_adr+2] = 0.01
        
        # ramdomize T_block quat
        rad = np.random.uniform(-0.5, 0.5)
        self.data.qpos[self.t_qpos_adr+3 : self.t_qpos_adr+7] = [np.cos(rad), 0.0, 0.0, np.sin(rad)]
        
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation(), {}

    def step(self, action):
        """
        action: [target_x, target_y]
        """
        self._set_mocap_2d(action)

        # step sim
        control_dt = 1.0 / 10.0 #fps=10
        sim_steps = int(control_dt / self.model.opt.timestep)
        
        for _ in range(sim_steps):
            self.data.ctrl[self.act_ids] = self.data.qpos[self.act_ids]
            mujoco.mj_step(self.model, self.data)

        obs = self.get_observation()
        
        ok, dxy, dyaw = check_xy_pose_match(
            self.model, self.data, "T_sign_anchor", "T_block_anchor", 
            pos_tol=0.015, yaw_tol_deg=5.0
        )
        
        # reward function
        reward = -0.01 
        if ok:
            reward += 20.0
            
        self.current_step += 1
        terminated = ok
        truncated = self.current_step >= self.max_steps
        
        info = {"dxy": dxy, "dyaw": dyaw}
        
        return obs, reward, terminated, truncated, info

    def render(self):
        img1 = cv2.resize(self._obs["cam_top"], (448, 448))
        img2 = cv2.resize(self._obs["cam_side"], (448, 448))
        img = np.concatenate([img1, img2], axis=1)
        return img
    
    def close(self):
        return super().close()
