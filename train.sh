#!/bin/bash

# 設置參數
MODEL_TYPE="tiny"                                   # 模型類型（tiny 或 small）
PKL_PATH="./MSWC_MIN_10.pkl"                        # 原始數據的路徑
CHECKPOINT_DIR="./checkpoints"                      # 檢查點的主資料夾
CHECKPOINT_NAME="20250217_161556/epoch_10.pt"                # 初始檢查點的名稱
BATCH_SIZE=8                                        # DataLoader 的批次大小
INPUT_DIM=40                                        # 輸入特徵維度
VIRTUAL_BATCH_SIZE=1000                             # 虛擬批次大小
ENCODER_DIM=256                                     # 編碼器維度
NUM_ENCODER_LAYERS=4                                # 編碼器層數
NUM_ATTENTION_HEADS=4                               # 注意力頭數
EPOCHS=10                                            # 訓練的 epoch 數量
LEARNING_RATE=5e-4                                  # 優化器的學習率
ALPHA=8
BETA=-4
NOISE_PROBABILITY=0.0

# 創建以時間命名的子資料夾（例如：checkpoints/20250107_101530）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="${CHECKPOINT_DIR}/${TIMESTAMP}"
mkdir -p "$SAVE_DIR"

# 訓練日誌文件（與檢查點存放在同一資料夾）
LOG_FILE="${SAVE_DIR}/training.log"

# 運行訓練腳本並保存日誌
python train.py \
    --model_type "$MODEL_TYPE" \
    --pkl_path "$PKL_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --checkpoint_name "$CHECKPOINT_NAME" \
    --save_dir "$SAVE_DIR" \
    --batch_size $BATCH_SIZE \
    --input_dim $INPUT_DIM \
    --virtual_length $VIRTUAL_BATCH_SIZE \
    --encoder_dim $ENCODER_DIM \
    --num_encoder_layers $NUM_ENCODER_LAYERS \
    --num_attention_heads $NUM_ATTENTION_HEADS \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --alpha_beta $ALPHA $BETA \
    --noise_prob $NOISE_PROBABILITY \
    | tee "$LOG_FILE"

echo "訓練完成！檢查點和日誌已保存至 $SAVE_DIR"
