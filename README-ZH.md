# MuJoCo SO100 PushT：Diffusion Policy 实现

本仓库提供了在 **MuJoCo** 物理引擎中使用 **SO100 机械臂** 完成 **PushT 任务** 的完整流程。项目包含与 Gymnasium 兼容的环境、用于数据采集的遥操作接口，以及遵循 **LeRobot 4.4** 生态系统的训练/推理工作流。

![alt text](assets/image.png)

## 📂 项目结构

```bash
├── chernyadev/               # 资源文件与 MJCF 模型
│   └── ... /trs_so_arm100    # SO100 场景配置（scene.xml, test_env.xml）
├── data/                     # 数据集存储
│   ├── NewData*/             # 处理后的演示数据集
├── script/                   # 自动化脚本工具
│   ├── record_demonstration_data.sh # 批量数据采集脚本
│   ├── infer.sh              # 策略评估脚本
│   └── train_policy.sh       # 策略训练脚本
├── src/                      # 核心源代码
│   ├── env_human_*.py        # 遥操作接口（末端/伺服模式）用于数据采集
│   ├── env_gym_*.py          # 用于训练/推理的 Gymnasium 封装环境
│   ├── train.py              # Diffusion Policy 训练流程（基于 LeRobot 4.0）
│   ├── infer.py              # 模型评估与推理测试
│   └── helper.py             # 通用工具函数与环境辅助模块
└── outputs/                  # 训练与评估结果输出
    ├── ckpt/                 # 模型检查点
    ├── runs/                 # TensorBoard 日志与训练指标
    └── recorded_videos/      # 策略评估episode的渲染视频
```

---

## 🚀 工作流程指南

### 1. 环境配置

本项目依赖 `mujoco`、`gymnasium` 和 `lerobot`（v4.4）库。

运行以下命令创建康达环境：
```bash
conda env create -f environment.yml
```

### 2. 人类演示数据采集

使用 `src/env_human_ee.py` 通过游戏手柄（如 Xbox/PS5）采集高质量演示数据。该脚本将摇杆输入映射为 MuJoCo 中的**末端执行器（EE）增量位置**。

![image-20260314190824852](assets/image-20260314190824852.png)

```bash
./script/record_demonstration_data.sh
```

### 3. 数据处理（LeRobot 4.4）

采集到的数据会被转换为 **LeRobot 数据集格式**（Zarr/Parquet），以确保与现代模仿学习流程兼容。此过程包括生成 Diffusion Policy 所需的元数据。

你可以使用 `lerobot-dataset-viz` 可视化数据集：

```bash
lerobot-dataset-viz --repo-id <你的数据路径> --episode-index 12
```

此外，我已将数据集上传至 Hugging Face。如果你不想自行采集数据，可在此下载：[qian1dqs/so100-pusht](https://huggingface.co/datasets/qian1dqs/so100-pusht)

### 4. 策略训练

我们采用基于 CNN 的 **Diffusion Policy** 来学习 PushT 任务的多模态动作分布。

```bash
./script/train_policy.sh
```

> 我在 A100 上训练了 1000 个 epoch。模型也已上传至 [Hugging Face](https://huggingface.co/qian1dqs/so100-pusht-diffusion)。

损失曲线：

![image-20260314191742736](assets/image-20260314191742736.png)

### 5. 评估与推理

运行 `src/infer.py` 加载训练好的检查点，在 MuJoCo 仿真环境中测试策略性能。该脚本提供实时渲染以可视化智能体行为。

```bash
./script/infer.sh
```

以下是两个推理示例：
![PushT Task Demo1](assets/show1.gif)
![PushT Task Demo2](assets/show2.gif)

---

## 🔧 常见问题排查

- **MuJoCo 渲染显示问题**：确保系统具备可用的 OpenGL 上下文。在无头服务器上使用时，请在 `mujoco.Renderer` 中设置 `headless=True`。
- **LeRobot 版本不匹配**：本项目基于 `lerobot==4.4` 测试。其他版本可能需要调整 API 调用。
- **数据集加载错误**：请确认数据路径中包含必需的 `meta.json` 文件和 Zarr 数据块。

---

## 📄 许可证

本项目采用 MIT 许可证发布。详情请参见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [LeRobot](https://github.com/huggingface/lerobot) 提供的模仿学习框架
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) 提供的策略架构
- [MuJoCo](https://mujoco.readthedocs.io/) 提供的物理仿真引擎

---

> 💡 **提示**：为获得最佳效果，建议采集至少 200 条多样化的演示 episode。在 PushT 这类接触丰富的任务中，数据质量对策略性能影响显著。