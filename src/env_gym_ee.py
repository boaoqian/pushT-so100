import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import cv2
from scipy.spatial.transform import Rotation as R
from helper import *

class PushT(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, xml_path, max_steps=1000, render_mode=None):
        super(PushT, self).__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.pos_random_range = 0.05
        
        # 加载 MuJoCo 模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=224, width=224)

        # 1. 配置 Action Space: [x, y,
        self.action_space = spaces.Box(
            low=np.array([-1, -1]), 
            high=np.array([1, 1]), 
            dtype=np.float32
        )

        # 2. 配置 Observation Space
        self.observation_space = spaces.Dict({
            "cam_top": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            "cam_side": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            "observation.state": spaces.Box(low=-10.0, high=10.0, shape=(5,), dtype=np.float32),
        })

        # 获取 ID
        self.act_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
        self.act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.act_names]
        self.mocap_id = self.model.body("target_mocap").mocapid[0]
        
        t_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "T_block")
        self.t_qpos_adr = self.model.jnt_qposadr[self.model.body_jntadr[t_body_id]]

        self.current_step = 0
        self.key_id = 0
        self._obs = None

    def _set_mocap_4d(self, action):
            """ 
            将 Action 应用于 Mocap 目标 
            action: [x,y]
            """
            dx, dy = action
            org_pos = self.data.mocap_pos[self.mocap_id]
            self.data.mocap_pos[self.mocap_id] = [dx, dy, org_pos[2]]
    def get_observation(self):
        # 视觉渲染
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
        # 重置到初始关键帧
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        
        # 随机化 T_block 位置
        self.data.qpos[self.t_qpos_adr : self.t_qpos_adr+2] = [
            np.random.uniform(0.25 - self.pos_random_range, 0.25+self.pos_random_range),
            np.random.uniform(-self.pos_random_range, self.pos_random_range)
        ]
        self.data.qpos[self.t_qpos_adr+2] = 0.01
        
        # 随机化 T_block 角度
        rad = np.random.uniform(-0.5, 0.5)
        self.data.qpos[self.t_qpos_adr+3 : self.t_qpos_adr+7] = [np.cos(rad), 0.0, 0.0, np.sin(rad)]
        
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation(), {}

    def step(self, action):
        """
        action: [target_x, target_y]
        """
        # 1. 应用 Action 到 Mocap
        self._set_mocap_4d(action)

        # 2. 物理仿真推进
        control_dt = 1.0 / 10.0 # 对应 render_fps
        sim_steps = int(control_dt / self.model.opt.timestep)
        
        for _ in range(sim_steps):
            # 保持执行器目标等于当前关节位置（位置伺服控制）
            self.data.ctrl[self.act_ids] = self.data.qpos[self.act_ids]
            mujoco.mj_step(self.model, self.data)

        # 3. 计算奖励与状态
        obs = self.get_observation()
        
        # 检查是否成功 (调用 helper.py 中的函数)
        ok, dxy, dyaw = check_xy_pose_match(
            self.model, self.data, "T_sign_anchor", "T_block_anchor", 
            pos_tol=0.015, yaw_tol_deg=5.0
        )
        
        # 基础奖励逻辑
        reward = -0.01 # 时间惩罚
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

# --- 测试脚本 ---
if __name__ == "__main__":
    from gymnasium.wrappers import RecordVideo
    XML_PATH = "./chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/human_env.xml"
    # 1. 实例化环境并指定渲染模式
    raw_env = PushT(XML_PATH, render_mode="rgb_array")
    
    env = RecordVideo(
        raw_env, 
        video_folder="outputs/recorded_videos", 
        episode_trigger=lambda x: True,
        name_prefix="pusht_video"
    )

    obs, _ = env.reset()
    
    try:
        for i in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            frame = cv2.cvtColor(obs["cam_top"], cv2.COLOR_RGB2BGR)
            cv2.imshow("PushT Live Preview", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if terminated or truncated:
                obs, _ = env.reset()
                print(f"Episode finished, video saved.")
                
    finally:
        env.close()
        cv2.destroyAllWindows()