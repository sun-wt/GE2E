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
  --model_type "conformer" \
  --checkpoint_path "./checkpoints/40_256_2_2.pt" \
  --input_dim 40 \
  --encoder_dim 256 \
  --num_encoder_layers 2 \
  --num_attention_heads 2 \
  | tee "$results_dir/evaluation.log"

# 確認是否有生成 DET 曲線資料夾，並移動到結果資料夾
if [ -d "det_curves" ]; then
    mv "det_curves" "$results_dir/"
    echo "每個字的 DET 曲線已保存至 $results_dir/det_curves/"
fi

# 確認綜合 DET 曲線是否生成，並移動到結果資料夾
if [ -f "overall_det_curve.png" ]; then
    mv "overall_det_curve.png" "$results_dir/"
    echo "綜合 DET 曲線已保存至 $results_dir/overall_det_curve.png"
else
    echo "綜合 DET 曲線未生成或未找到。"
fi
