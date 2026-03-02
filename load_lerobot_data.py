import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "/media/qba/Data/Project/Robot/So100PushT/data/myDataset2"

dataset = LeRobotDataset(repo_id)

print(dataset)
# 2) Random access by index
sample = dataset[0]
print(sample)
