#!/bin/bash

# 設置參數
PKL_PATH="generate/train_fixed_35_1600.pkl"          # 訓練數據的路徑
CHECKPOINT_DIR="./checkpoints"                    # 檢查點的主資料夾
LOGS_DIR="./logs"                # 訓練日誌的資料夾
CHECKPOINT_NAME="h0.pt"                           # 初始檢查點的名稱
BATCH_SIZE=1                                      # DataLoader 的批次大小
SHUFFLE=True                                      # 是否打亂數據
INPUT_DIM=80                                      # 輸入特徵維度
ENCODER_DIM=128                                   # 編碼器維度
NUM_ENCODER_LAYERS=2                              # 編碼器層數
NUM_ATTENTION_HEADS=4                             # 注意力頭數
EPOCHS=20                                         # 訓練的 epoch 數量
LEARNING_RATE=1e-3                                # 優化器的學習率

# 創建 logs 資料夾（如果不存在）
mkdir -p "$LOGS_DIR"

# 訓練日誌文件
LOG_FILE="${LOGS_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

# 運行訓練腳本並保存日誌
python fix.py \
    --pkl_path "$PKL_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --checkpoint_name "$CHECKPOINT_NAME" \
    --batch_size $BATCH_SIZE \
    --shuffle $SHUFFLE \
    --input_dim $INPUT_DIM \
    --encoder_dim $ENCODER_DIM \
    --num_encoder_layers $NUM_ENCODER_LAYERS \
    --num_attention_heads $NUM_ATTENTION_HEADS \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    | tee "$LOG_FILE"

echo "訓練完成。日誌已保存至 $LOG_FILE"