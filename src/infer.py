import argparse
import logging
from pathlib import Path

import torch
import cv2
from env_gym_ee import PushT
from gymnasium.wrappers import RecordVideo
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run diffusion policy inference for PushT")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="outputs/ckpt/final_model",
        help="Path to the pretrained checkpoint directory"
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="data/NewData3.9-ee-2d-pos",
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--env_path",
        type=str,
        default="chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/human_env.xml",
        help="Path to the MuJoCo XML environment file"
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default="outputs/recorded_videos",
        help="Folder to save evaluation videos"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.ckpt_path)
    dataset_id = Path(args.dataset_id)
    env_path = args.env_path
    video_folder = args.video_folder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained policy
    policy = DiffusionPolicy.from_pretrained(ckpt_path.absolute())
    policy.eval()
    policy.to(device)

    # Load dataset metadata and build preprocessors
    dataset_metadata = LeRobotDatasetMetadata(dataset_id.absolute())
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        dataset_stats=dataset_metadata.stats,
        pretrained_path=ckpt_path,
    )

    # Create environment
    raw_env = PushT(xml_path=env_path, render_mode="rgb_array")
    env = RecordVideo(
        raw_env,
        video_folder=video_folder,
        episode_trigger=lambda x: True,
        name_prefix="pusht_eval_video"
    )

    obs, _ = env.reset()

    logger.info("Starting inference...")
    terminated = False
    truncated = False

    try:
        while not terminated and not truncated:
            # Show live preview
            frame = cv2.cvtColor(obs["cam_top"], cv2.COLOR_RGB2BGR)
            cv2.imshow("PushT Live Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Run policy inference
            with torch.no_grad():
                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_metadata.features,
                    device=device,
                )
                obs_tensor = preprocess(obs_frame)
                actions_sequence = policy.select_action(obs_tensor)
                actions_sequence = postprocess(actions_sequence)

            # Execute the first action in the predicted sequence
            actions_to_execute = actions_sequence[0].cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(actions_to_execute)

            if terminated and not truncated:
                logger.info(f"Target reached! {info}")

    except KeyboardInterrupt:
        logger.info("Inference interrupted by user.")
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()