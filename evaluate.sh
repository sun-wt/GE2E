#!/bin/bash

# 設置參數
ENROLL_PATH="./enroll.pkl"                     # Enroll 數據的路徑
TEST_PATH="./test.pkl"                         # Test 數據的路徑
CHECKPOINT_PATH="./checkpoints/e0.pt"          # 模型檢查點的路徑
INPUT_DIM=80                                   # 輸入特徵維度（需與訓練時一致）
ENCODER_DIM=128                                # 編碼器維度（需與訓練時一致）
NUM_ENCODER_LAYERS=2                           # 編碼器層數（需與訓練時一致）
NUM_ATTENTION_HEADS=4                          # 注意力頭數（需與訓練時一致）
NUM_CLASSES=35                                 # 分類數（需與訓練時一致）

# 創建結果保存的資料夾
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="./results/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# 日誌文件
LOG_FILE="${RESULTS_DIR}/evaluation.log"

# 運行評估腳本
python evaluate.py \
    --enroll_path "$ENROLL_PATH" \
    --test_path "$TEST_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --input_dim $INPUT_DIM \
    --encoder_dim $ENCODER_DIM \
    --num_encoder_layers $NUM_ENCODER_LAYERS \
    --num_attention_heads $NUM_ATTENTION_HEADS \
    --num_classes $NUM_CLASSES \
    | tee "$LOG_FILE"

echo "評估完成！結果已保存至 $RESULTS_DIR"

mkdir -p "result/$(date +%Y%m%d_%H%M%S)" && python evaluate.py --enroll_path "./enroll.pkl" --test_path "./test.pkl" --checkpoint_path "./checkpoints/e0.pt" --input_dim 80 --encoder_dim 128 --num_encoder_layers 2 --num_attention_heads 4 --num_classes 35 | tee "result/$(date +%Y%m%d_%H%M%S)/evaluation.log"
python evaluate.py --enroll_path "./enroll.pkl" --test_path "./test.pkl" --checkpoint_path "./checkpoints/20250109_100135/epoch_2.pt" --input_dim 80 --encoder_dim 128 --num_encoder_layers 2 --num_attention_heads 4 --num_classes 35 --keywords "house,wow,dog,no,up,marvin,follow,backward,nine,off,one,seven,bed,learn,sheila,down,two,left,zero,six,cat,eight,four,three,stop,tree,bird"

python evaluate.py --enroll_path "./enroll.pkl" --test_path "./test.pkl" --checkpoint_path "./checkpoints/20250113_163443/epoch_5.pt" --input_dim 80 --encoder_dim 128 --num_encoder_layers 2 --num_attention_heads 4 --num_classes 35 
