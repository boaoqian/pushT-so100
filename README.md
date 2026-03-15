# MuJoCo SO100 PushT: Diffusion Policy Implementation

This repository provides a complete pipeline for the **PushT task** using the **SO100 robotic arm** within the **MuJoCo** physics engine. It features a Gymnasium-compatible environment, teleoperation for data collection, and training/inference workflows following the **LeRobot 4.4** ecosystem.

[中文版md](README-ZH.md)

![alt text](assets/image.png)

## 📂 Project Structure

```bash
├── chernyadev/               # Assets and MJCF models
│   └── ... /trs_so_arm100    # Specific SO100 scene configurations (scene.xml, test_env.xml)
├── data/                     # Dataset storage
│   ├── NewData*/             # Processed demonstration datasets
├── script/                   # Automation utilities
│   ├── record_demonstration_data.sh # Script for batch data collection
│   ├── infer.sh              # Script for policy evaluation
│   └── train_policy.sh       # Script for training policy
├── src/                      # Core source code
│   ├── env_human_*.py        # Teleop interfaces (EE/Servo) for data collection
│   ├── env_gym_*.py          # Gymnasium wrappers for training/inference
│   ├── train.py              # Diffusion Policy training pipeline (LeRobot 4.0)
│   ├── infer.py              # Model evaluation and inference testing
│   └── helper.py             # Common utilities and environment helpers
└── outputs/                  # Training and evaluation results
    ├── ckpt/                 # Model checkpoints
    ├── runs/                 # TensorBoard logs and training metrics
    └── recorded_videos/      # Renders of policy evaluation episodes
```

---

## 🚀 Workflow Guide

### 1. Environment Setup

The project relies on `mujoco`, `gymnasium`, and the `lerobot` (v4.4) library. 

Run `conda env create -f environment.yml` to set up the environment.

### 2. Human Demonstration Collection

Use `src/env_human_ee.py` to collect high-quality demonstrations via a game controller (e.g., Xbox/PS5). This script maps joystick input to **End-Effector (EE)** delta positions in MuJoCo.

![image-20260314190824852](assets/image-20260314190824852.png)

```bash
./script/record_demonstration_data.sh
```

### 3. Data Processing (LeRobot 4.4)

The collected data is converted into the **LeRobot dataset format** (Zarr/Parquet), ensuring compatibility with modern imitation learning pipelines. This includes generating the necessary metadata for the Diffusion Policy.

You can use `lerobot-dataset-viz` to visualize your dataset like this:

```bash
lerobot-dataset-viz --repo-id <your-data-path> --episode-index 12
```

Additionally, I have uploaded my dataset to Hugging Face. If you prefer not to collect data yourself, you can download it here: [qian1dqs/so100-pusht](https://huggingface.co/datasets/qian1dqs/so100-pusht)

### 4. Policy Training

We utilize the **Diffusion Policy** (CNN-based) to learn the multimodal distribution of the PushT task.

```bash
./script/train_policy.sh
```

I trained this model on an A100 for 1000 epochs. The model is also available on [Hugging Face](https://huggingface.co/qian1dqs/so100-pusht-diffusion).

Loss curve:

![image-20260314191742736](assets/image-20260314191742736.png)

### 5. Evaluation & Inference

Run `src/infer.py` to load a trained checkpoint and test its performance in the MuJoCo simulation. The script provides real-time rendering to visualize the agent's behavior.

```bash
./script/infer.sh
```
Here are two examples of inference:
![PushT Task Demo1](assets/show1.gif)
![PushT Task Demo2](assets/show2.gif)

---

## 🔧 Troubleshooting

- **Display issues with MuJoCo rendering**: Ensure you have a working OpenGL context. For headless servers, use `mujoco.Renderer` with `headless=True`.
- **LeRobot version mismatch**: This project is tested with `lerobot==4.4`. Other versions may require API adjustments.
- **Dataset loading errors**: Verify that your data path contains the required `meta.json` and Zarr chunks.

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [LeRobot](https://github.com/huggingface/lerobot) for the imitation learning framework
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) for the policy architecture
- [MuJoCo](https://mujoco.readthedocs.io/) for the physics simulation

---

> 💡 **Tip**: For best results, collect at least 200 diverse demonstration episodes. Data quality significantly impacts policy performance in contact-rich tasks like PushT.

