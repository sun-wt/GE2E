import os
import warnings
import tensorflow as tf
import numpy as np

from conformer import build_conformer_encoder
from dataset import libriphrase, mswc
from criterion.GE2E_loss import GE2ELoss

# 環境設置
os.environ["TF_TRT_ENABLE"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 僅顯示錯誤信息
os.environ["TF_XLA_FLAGS"] = '--tf_xla_enable_xla_devices=0'

warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# 設定隨機種子
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# GPU 設置
print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs available: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPU available. Running on CPU.")

# 超參數
GLOBAL_BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3

def convert_sequence_to_tf_dataset(dataloader):
    def generator():
        for batch in dataloader:
            inputs = batch[0]
            labels = batch[1]
            yield inputs, tf.cast(labels, tf.float32)

    output_signature = (
        tf.TensorSpec(shape=(None, None, dataloader.input_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)

train_dataloader = mswc.MSWCDataloader(
        batch_size=GLOBAL_BATCH_SIZE,
        input_dim=80,
        train=True,
        shuffle=True
    )

train_dataset = convert_sequence_to_tf_dataset(train_dataloader)

# 創建 Conformer 模型
conformer_encoder = build_conformer_encoder(
    input_dim=80,            # 假設音頻特徵維度為 80
    num_layers=16,           # 堆疊的 Conformer Block 數量
    embed_dim=80,            # 嵌入維度
    num_heads=8,             # 注意力頭數
    ffn_expansion_factor=4,  # 前饋網路擴展因子
    conv_kernel_size_list=[3, 5, 7],  # 使用多個卷積核尺寸
    dropout_rate=0.1         # 增加 Dropout 避免過擬合
)


# 打印模型結構
conformer_encoder.summary()

# 設置損失和優化器
loss_fn = GE2ELoss()
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# 性能評估函數
def evaluate_model(dataset):
    total_loss = 0.0
    step = 0  # 確保 step 初始化
    for step, (inputs, labels) in enumerate(dataset):
        embeddings = conformer_encoder(inputs, training=False)
        loss = loss_fn(embeddings, labels)
        total_loss += loss.numpy()
    if step == 0:  # 如果沒有數據，返回 0 或其他合理的預設值
        return 0.0
    return total_loss / (step + 1)


# 訓練循環
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    total_loss = 0.0

    # 訓練階段
    for step, (inputs, labels) in enumerate(train_dataset):
        # print(f"Batch {step}:")
        # print(f"Inputs shape: {inputs.shape}")  # 輸入特徵
        # print(f"Labels shape: {labels.shape}")  # 標籤

        with tf.GradientTape() as tape:
            embeddings = conformer_encoder(inputs, training=True)
            loss = loss_fn(embeddings, labels)

        # 梯度更新
        gradients = tape.gradient(loss, conformer_encoder.trainable_weights)
        gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]  # 梯度裁剪
        optimizer.apply_gradients(zip(gradients, conformer_encoder.trainable_weights))

        total_loss += loss.numpy()
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.numpy():.4f}")

    # 每個 epoch 結束後進行評估
    train_loss = evaluate_model(train_dataset)
    # test_loss = evaluate_model(test_dataset)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

    # 保存模型權重
    checkpoint_dir = './checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_weights.h5')
    conformer_encoder.save_weights(checkpoint_path)
    print(f"Model weights saved to: {checkpoint_path}")

print("\nTraining complete. Model weights saved.")
