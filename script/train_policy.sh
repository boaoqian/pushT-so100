export CUDA_VISIBLE_DEVICES=0
python src/train.py \
  --data-path ./data/NewData3.9-ee-2d-pos \
  --output-dir outputs/pusht_diffusion \
  --batch-size 16 \
  --training-steps 13000 \
  --warmup-steps 1000 \
  --log-freq 100 \
  --save-freq 1000 \
  --lr 1e-4 \
  --weight-decay 1e-6 \
  --num-workers 4 \
  --n-obs-steps 2 \
  --horizon 16 \
  --n-action-steps 6 \
  --vision-backbone resnet18 \
  --device cuda