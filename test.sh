#!/bin/bash

# 創建時間戳資料夾
timestamp=$(date +"%Y%m%d_%H%M%S")
results_dir="results/$timestamp"
mkdir -p $results_dir

# 指令執行與記錄
python test.py \
  --enroll_path "./DB/enroll.pkl" \
  --test_path "./DB/test.pkl" \
  --model_type "conformer" \
  --checkpoint_path "./checkpoints/20250117_113747/epoch_2.pt" \
  --input_dim 40 \
  --encoder_dim 256 \
  --num_encoder_layers 2 \
  --num_attention_heads 2 \
#   --keywords "right" \
  | tee "$results_dir/evaluation.log"
