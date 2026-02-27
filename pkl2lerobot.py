import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def batch_convert_pkls_to_lerobot(input_dir, repo_id, fps=30, tolerance=0.01):
    input_path = Path(input_dir)
    pkl_files = sorted(list(input_path.glob("*.pkl")))
    
    if not pkl_files:
        print(f"在 {input_dir} 中没找到 pkl 文件！")
        return

    # 读取第一个文件初始化结构
    with open(pkl_files[0], "rb") as f:
        first_buffer = pickle.load(f)
    
    c, h, w = first_buffer["cam_top"][0].shape[2], first_buffer["cam_top"][0].shape[0], first_buffer["cam_top"][0].shape[1]
    state_dim = first_buffer["pose"][0].shape[0]
    action_dim = first_buffer["action"][0].shape[0]

    # 3. 创建 LeRobot 数据集实例
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="my_robot",
        features={
            "observation.images.cam_top": {
                "dtype": "video", 
                "shape": (c, h, w), 
                "names": ["channels", "height", "width"]
            },
            "observation.images.cam_side": {
                "dtype": "video", 
                "shape": (c, h, w), 
                "names": ["channels", "height", "width"]
            },
            "observation.state": {"dtype": "float32", "shape": (state_dim,)},
            "action": {"dtype": "float32", "shape": (action_dim,)}
        }
    )

    print(f"开始转换，容差设定为: {tolerance}")
    total_skipped = 0

    for pkl_file in tqdm(pkl_files, desc="Converting Episodes"):
        with open(pkl_file, "rb") as f:
            buffer = pickle.load(f)
        
        num_frames = len(buffer["cam_top"])
        last_action = None
        episode_frames_added = 0
        
        for i in range(num_frames):
            current_action = buffer["action"][i].astype(np.float32)

            # --- 使用 np.allclose 进行容差判断 ---
            # atol 表示绝对误差 (Absolute Tolerance)
            if last_action is not None and np.allclose(current_action, last_action, atol=tolerance):
                total_skipped += 1
                continue
            
            last_action = current_action
            # ----------------------------------

            img_top = torch.from_numpy(buffer["cam_top"][i]).permute(2, 0, 1)
            img_side = torch.from_numpy(buffer["cam_side"][i]).permute(2, 0, 1)

            frame_data = {
                "observation.images.cam_top": img_top,
                "observation.images.cam_side": img_side,
                "observation.state": torch.from_numpy(buffer["pose"][i].astype(np.float32)),
                "action": torch.from_numpy(current_action),
                "task" : "pushT"
            }
            
            dataset.add_frame(frame_data)
            episode_frames_added += 1
        
        if episode_frames_added > 0:
            dataset.save_episode()

    print(f"\n转换完成！总计跳过近似重复帧: {total_skipped}")

if __name__ == "__main__":
    INPUT_FOLDER = "RecordTemp"
    OUTPUT_REPO = "/media/qba/Data/Project/Robot/So100PushT/myDataset"
    # 调用时可以自定义 tolerance
    batch_convert_pkls_to_lerobot(INPUT_FOLDER, OUTPUT_REPO, tolerance=0.01)