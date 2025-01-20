#!/bin/bash

# 創建帶時間戳的結果資料夾
timestamp=$(date +"%Y%m%d_%H%M%S")
results_dir="results/$timestamp"
mkdir -p $results_dir

# 執行指令，並記錄輸出到結果資料夾
python test.py \
  --data_dir "/datas/store162/syt/GE2E/DB/google_speech_commands" \
  --results_dir $results_dir \
  --enroll_path "./DB/enroll.pkl" \
  --test_path "./DB/test.pkl" \
  --model_type "tiny" \
  --checkpoint_path "./checkpoints/20250118_181302/epoch_9.pt" \
  --input_dim 40 \
  --encoder_dim 512 \
  --num_encoder_layers 12 \
  --num_attention_heads 8 \
  | tee "$results_dir/evaluation.log"

det_curves_path=$(find . -type d -name "det_curves" | head -n 1)
if [ -d "$det_curves_path" ]; then
    mv "$det_curves_path" "$results_dir/"
    echo "每個字的 DET 曲線已保存至 $results_dir/det_curves/"
else
    echo "DET 曲線資料夾未生成或未找到。"
fi

overall_det_curve_path=$(find . -type d -name "overall_det_curve.png" | head -n 1)
if [ -d "$overall_det_curve_path" ]; then
    mv "$overall_det_curve_path" "$results_dir/"
    echo "每個字的 DET 曲線已保存至 $results_dir/overall_det_curve.png"
else
    echo "DET 曲線資料夾未生成或未找到。"
fi
