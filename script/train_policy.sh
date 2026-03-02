lerobot-train \
    --policy.type diffusion \
    --dataset.repo_id /media/qba/Data/Project/Robot/So100PushT/data/pusht_so100_dataset_merge \
    --policy.device cuda \
    --batch_size 1 \
    --steps 100000 \
    --policy.repo_id /media/qba/Data/Project/Robot/So100PushT/model/pusht_so100_diffusion_model