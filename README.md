# MuJoCo SO100 PushT: Diffusion Policy Implementation

This repository provides a complete pipeline for the **PushT task** using the **SO100 robotic arm** within the **MuJoCo** physics engine. It features a Gymnasium-compatible environment, teleoperation for data collection, and training/inference workflows following the **LeRobot 4.0** ecosystem.

## 📂 Project Structure

```bash
├── chernyadev/               # Assets and MJCF models
│   └── ... /trs_so_arm100    # Specific SO100 scene configurations (scene.xml, test_env.xml)
├── data/                     # Dataset storage
│   ├── NewData*/             # processed demonstration  datasets
├── script/                   # Automation utilities
│   ├── record_data.sh        # Script for batch data collection
│   ├── merge_dataset.sh      # Script for dataset consolidation
│   └── train_policy.sh       # Script for launching training jobs
├── src/                      # Core source code
│   ├── env_human_*.py        # Teleop interfaces (EE/Servo) for data collection
│   ├── env_gym_*.py          # Gymnasium wrappers for training/inference
│   ├── train.py              # Diffusion Policy training pipeline (LeRobot 4.0)
│   ├── eval_policy.py        # Model evaluation and inference testing
│   └── helper.py             # Common utilities and environment helpers
└── outputs/                  # Training and evaluation results
    ├── ckpt/                 # Model checkpoints (.pt files)
    ├── runs/                 # TensorBoard logs and training metrics
    └── recorded_videos/      # Renders of policy evaluation episodes

```

---

## 🚀 Workflow Guide

### 1. Environment Setup

The project relies on `mujoco`, `gymnasium`, and the `lerobot` (v4.0+) library. 
### 2. Human Demonstration Collection

Use `src/env_human_ee.py` to collect high-quality demonstrations via a game controller (e.g., Xbox/PS5). This script maps joystick input to the **End-Effector (EE)** delta positions in MuJoCo.



```bash
python src/env_human_ee.py --num_episodes 50 --output_dir ./data/raw_demos

```

### 3. Data Processing (LeRobot 4.0)

The collected data is converted into the **LeRobot dataset format** (Zarr/Parquet), ensuring compatibility with modern imitation learning pipelines. This includes generating the necessary metadata for the Diffusion Policy.

### 4. Policy Training

We utilize the **Diffusion Policy** (CNN or Transformer-based) to learn the multimodal distribution of the PushT task.

```bash
python src/train.py \
    --policy.type=diffusion \
    --dataset.path=./data/lerobot_dataset \
    --device=cuda

```

### 5. Evaluation & Inference

Run `src/eval_policy.py` to load a trained checkpoint and test its performance in the MuJoCo simulation. The script provides real-time rendering to visualize the agent's behavior.

```bash
python src/eval_policy.py --checkpoint ./outputs/checkpoints/last.pt

```
