# Merge train and validation splits back into one dataset
lerobot-edit-dataset \
    --repo_id /media/qba/Data/Project/Robot/So100PushT/data/merge_dataset \
    --operation.type merge \
    --operation.repo_ids "[
    '/media/qba/Data/Project/Robot/So100PushT/data/NewData3.4-easy',
    '/media/qba/Data/Project/Robot/So100PushT/data/NewData3.5-new-init'
    ]"