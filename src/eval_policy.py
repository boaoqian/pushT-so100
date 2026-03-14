from pathlib import Path
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import math
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

DATA_PATH = "/home/baqian/qba/pusht/NewData3.9-ee-2d-pos"
# --- 训练参数 ---
BATCH_SIZE = 16
training_steps = 13000
WARMUP_STEPS = 1000
log_freq = 100
save_freq = 1000

# ----------------
def lr_lambda(current_step: int):
    if current_step < WARMUP_STEPS:
        return float(current_step) / float(max(1, WARMUP_STEPS))
    progress = float(current_step - WARMUP_STEPS) / float(max(1, training_steps - WARMUP_STEPS))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
def main():
    output_directory = Path("outputs/pusht_diffusion")
    checkpoints_dir = output_directory / f"checkpoints_{time.strftime('%Y-%m-%d_%H:%M')}"
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_directory / f"runs_{time.strftime('%Y-%m-%d_%H:%M')}"))
    device = torch.device("cuda")
    
    
    dataset_metadata = LeRobotDatasetMetadata(DATA_PATH)
    features = dataset_to_policy_features(dataset_metadata.features)

    image_keys = ["observation.images.cam_top", "observation.images.cam_side"]
    mask_keys = []

    for key in image_keys:
        if key in features:
            features[key].shape = (3, 224, 224)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features  = {k: ft for k, ft in features.items() if k not in output_features and k not in mask_keys}

    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=2,
        horizon=16,
        n_action_steps=6,
        vision_backbone="resnet18"
    )

    delta_timestamps = {}
    for k in input_features.keys():
        delta_timestamps[k] = [i / dataset_metadata.fps for i in cfg.observation_delta_indices]
    for k in output_features.keys():
        delta_timestamps[k] = [i / dataset_metadata.fps for i in cfg.action_delta_indices]
    
    dataset = LeRobotDataset(DATA_PATH, delta_timestamps=delta_timestamps)

    policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )

    step = 0
    done = False
    print(f"Training started. Saving to {output_directory}")

    while not done:
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar("Loss/train", loss.item(), step)
                writer.add_scalar("LR/train", current_lr, step)
                print(f"step: {step} loss: {loss.item():.3f} lr: {current_lr:.6f}")

            if step > 0 and step % save_freq == 0:
                step_ckpt_dir = checkpoints_dir / f"step_{step}"
                step_ckpt_dir.mkdir(parents=True, exist_ok=True)
                
                policy.save_pretrained(step_ckpt_dir)
                preprocessor.save_pretrained(step_ckpt_dir)
                postprocessor.save_pretrained(step_ckpt_dir)
                print(f"Checkpoint saved at step {step}")

            step += 1
            if step >= training_steps:
                done = True
                break

    
    final_dir = output_directory / "checkpoints/final_model"
    policy.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)
    writer.close()
    print(f"Training finished. Final model saved to {final_dir}")

if __name__ == "__main__":
    main()