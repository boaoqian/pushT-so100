from pathlib import Path
import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train DiffusionPolicy on LeRobot dataset")

    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/baqian/qba/pusht/NewData3.9-ee-2d-pos",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/pusht_diffusion",
        help="Directory to save logs and checkpoints",
    )

    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--training-steps", type=int, default=13000, help="Total training steps")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--log-freq", type=int, default=100, help="Logging frequency")
    parser.add_argument("--save-freq", type=int, default=1000, help="Checkpoint save frequency")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")

    parser.add_argument("--n-obs-steps", type=int, default=2, help="Number of observation steps")
    parser.add_argument("--horizon", type=int, default=16, help="Prediction horizon")
    parser.add_argument("--n-action-steps", type=int, default=6, help="Number of action steps")
    parser.add_argument("--vision-backbone", type=str, default="resnet18", help="Vision backbone")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use, e.g. cuda or cpu",
    )

    parser.add_argument(
        "--image-keys",
        nargs="*",
        default=["observation.images.cam_top", "observation.images.cam_side"],
        help="Image feature keys",
    )
    parser.add_argument(
        "--mask-keys",
        nargs="*",
        default=[],
        help="Mask feature keys to exclude from input features",
    )

    return parser.parse_args()


def build_lr_lambda(warmup_steps: int, training_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return lr_lambda


def main():
    args = parse_args()

    output_directory = Path(args.output_dir)
    data_dir = Path(args.data_path)
    checkpoints_dir = output_directory / f"checkpoints_{time.strftime('%Y-%m-%d_%H:%M')}"
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_directory / f"runs_{time.strftime('%Y-%m-%d_%H:%M')}"))
    device = torch.device(args.device)

    dataset_metadata = LeRobotDatasetMetadata(data_dir.absolute())
    features = dataset_to_policy_features(dataset_metadata.features)

    image_keys = args.image_keys
    mask_keys = args.mask_keys

    for key in image_keys:
        if key in features:
            features[key].shape = (3, 224, 224)

    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features and k not in mask_keys}

    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=args.n_obs_steps,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        vision_backbone=args.vision_backbone,
    )

    delta_timestamps = {}
    for k in input_features.keys():
        delta_timestamps[k] = [i / dataset_metadata.fps for i in cfg.observation_delta_indices]
    for k in output_features.keys():
        delta_timestamps[k] = [i / dataset_metadata.fps for i in cfg.action_delta_indices]

    dataset = LeRobotDataset(data_dir.absolute(), delta_timestamps=delta_timestamps)

    policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        cfg, dataset_stats=dataset_metadata.stats
    )

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        build_lr_lambda(args.warmup_steps, args.training_steps),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    step = 0
    done = False
    print(f"Training started. Saving to {output_directory}")

    while not done:
        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % args.log_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar("Loss/train", loss.item(), step)
                writer.add_scalar("LR/train", current_lr, step)
                print(f"step: {step} loss: {loss.item():.3f} lr: {current_lr:.6f}")

            if step > 0 and step % args.save_freq == 0:
                step_ckpt_dir = checkpoints_dir / f"step_{step}"
                step_ckpt_dir.mkdir(parents=True, exist_ok=True)

                policy.save_pretrained(step_ckpt_dir)
                preprocessor.save_pretrained(step_ckpt_dir)
                postprocessor.save_pretrained(step_ckpt_dir)
                print(f"Checkpoint saved at step {step}")

            step += 1
            if step >= args.training_steps:
                done = True
                break

    final_dir = checkpoints_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    policy.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)

    writer.close()
    print(f"Training finished. Final model saved to {final_dir}")


if __name__ == "__main__":
    main()