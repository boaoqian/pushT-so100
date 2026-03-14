import torch
import numpy as np
import cv2
from pathlib import Path
from collections import deque

from env_gym_ee import PushT
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame
def main():
    ckpt_path = Path("outputs/final_model")
    dataset_id = Path("data/NewData3.9-ee-2d-pos")
    env_path = "chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/human_env.xml"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = DiffusionPolicy.from_pretrained(ckpt_path.absolute())
    policy.eval()
    policy.to(device)
    dataset_metadata = LeRobotDatasetMetadata(dataset_id.absolute())
    preprocess, postprocess = make_pre_post_processors(
        policy.config, dataset_stats=dataset_metadata.stats,pretrained_path=ckpt_path
    )

    # 3. 初始化环境
    env = PushT(xml_path=env_path ,render_mode="human")
    obs, _ = env.reset()

    print("开始推理...")
    terminated = False

    try:
        while not terminated:
            with torch.no_grad():
                obs_frame = build_inference_frame(
                observation=obs, ds_features=dataset_metadata.features, device=device
                )
                obs = preprocess(obs_frame)
                actions_sequence = policy.select_action(obs)
                actions_sequence = postprocess(actions_sequence)
            actions_to_execute = actions_sequence[0].numpy()
            obs, reward, terminated, truncated, info = env.step(actions_to_execute)
            env.render_cv2()

    except KeyboardInterrupt:
        print("\n停止推理。")
    finally:
        env.close()

if __name__ == "__main__":
    main()