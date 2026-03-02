# Merge train and validation splits back into one dataset
lerobot-edit-dataset \
    --repo_id /media/qba/Data/Project/Robot/So100PushT/data/merge_dataset \
    --operation.type merge \
    --operation.repo_ids "[
    '/media/qba/Data/Project/Robot/So100PushT/data/myDataset0',
    '/media/qba/Data/Project/Robot/So100PushT/data/myDataset1',
    '/media/qba/Data/Project/Robot/So100PushT/data/myDataset2',
    '/media/qba/Data/Project/Robot/So100PushT/data/myDataset3',
    '/media/qba/Data/Project/Robot/So100PushT/data/myDataset4']"