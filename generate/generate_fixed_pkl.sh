#!/bin/bash

# 設定變數
INPUT_PKL="/datas/store162/syt/GE2E/MSWC_MIN_10.pkl"  # 請替換為實際的輸入檔案路徑
OUTPUT_PKL="train_fixed_35_1600.pkl"  # 輸出的固定檔案路徑
BATCH_SIZE=8  # 每批次的單字數量
SAMPLES_PER_LABEL=35  # 每個單字的樣本數量
NUM_BATCHES=1600  # 批次數量

# 檢查輸入檔案是否存在
if [ ! -f "$INPUT_PKL" ]; then
  echo "錯誤: 找不到輸入檔案 $INPUT_PKL"
  exit 1
fi

# 執行 Python 腳本
echo "開始生成固定的 PKL 資料..."
python generate_fixed_pkl.py \
  --input_pkl "$INPUT_PKL" \
  --output_pkl "$OUTPUT_PKL" \
  --batch_size $BATCH_SIZE \
  --samples_per_label $SAMPLES_PER_LABEL \
  --num_batches $NUM_BATCHES

# 檢查輸出檔案是否成功生成
if [ -f "$OUTPUT_PKL" ]; then
  echo "成功生成固定的 PKL 檔案: $OUTPUT_PKL"
  echo "檢查單字選擇記錄檔案: ${OUTPUT_PKL%.pkl}_label_history.txt"
else
  echo "錯誤: 無法生成固定的 PKL 檔案"
  exit 1
fi
