import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from dataset.google import GoogleCommandsDataloader
from conformer import build_conformer_encoder
from pathlib import Path


# 設置隨機種子
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

# 加載預訓練模型
def load_conformer_model(weights_path, input_dim=80):
    conformer_encoder = build_conformer_encoder(
        input_dim=input_dim,
        num_layers=16,
        embed_dim=80,
        num_heads=8,
        ffn_expansion_factor=4,
        conv_kernel_size=31,
        dropout_rate=0.1
    )
    conformer_encoder.load_weights(weights_path)
    return conformer_encoder

def cosine_similarity(a, b, axis=-1):
    """
    計算餘弦相似度。
    """
    a_norm = tf.nn.l2_normalize(a, axis=axis)
    b_norm = tf.nn.l2_normalize(b, axis=axis)
    return tf.reduce_sum(a_norm * b_norm, axis=axis)

# 評估 GSC 數據集
def evaluate_gsc(conformer_encoder, dataloader, num_enrollment=10):
    dataloader.features = "g2p_embed"  # 使用 G2P 嵌入特徵
    dataset = convert_sequence_to_tf_dataset(dataloader)

    target_list = dataloader.target_list
    auc_scores = {}

    for target in target_list:
        print(f"Evaluating keyword: {target}")  # 打印當前目標單字

        # 選取 enroll 和 test 數據
        enroll_indices = dataloader.data[dataloader.data['keyword'] == target].index[:num_enrollment]
        test_indices = dataloader.data[dataloader.data['keyword'] == target].index[num_enrollment:]
        
        # 計算 enroll embeddings
        enroll_embeddings = []
        for i in enroll_indices:
            wav = dataloader.data.loc[i, 'wav']
            print(wav)
            enroll_audio = dataloader._load_wav(wav)
            enroll_audio = tf.expand_dims(enroll_audio, axis=0)  # 添加 batch 維度
            enroll_embeddings.append(
                conformer_encoder(enroll_audio, training=False)
            )
        print('centroid calculating')
        centroid = tf.reduce_mean(enroll_embeddings, axis=0, keepdims=True)

        print('positive_similarities calculating')
        # 測試數據 (正樣本)
        positive_similarities = []
        for i in test_indices:
            wav = dataloader.data.loc[i, 'wav']
            test_audio = dataloader._load_wav(wav)
            test_audio = tf.expand_dims(test_audio, axis=0)  # 添加 batch 維度
            test_embedding = conformer_encoder(test_audio, training=False)
            sim = tf.exp(cosine_similarity(test_embedding, centroid))  # 計算正樣本相似度
            positive_similarities.append(sim.numpy())  # 使用 `.numpy()` 轉換為純量

        print('negative_similarities calculating')

        # 測試數據 (負樣本)
        negative_similarities = []
        for other_target in target_list:
            if other_target != target:
                # 僅取最多 10 個樣本
                other_test_indices = dataloader.data[dataloader.data['keyword'] == other_target].index[:1]
                for i in other_test_indices:
                    wav = dataloader.data.loc[i, 'wav']
                    other_test_audio = dataloader._load_wav(wav)
                    other_test_audio = tf.expand_dims(other_test_audio, axis=0)  # 添加 batch 維度
                    other_test_embedding = conformer_encoder(
                        other_test_audio, training=False
                    )
                    neg_sim = tf.exp(cosine_similarity(other_test_embedding, centroid))  # 計算負樣本相似度
                    negative_similarities.append(neg_sim.numpy())  # 使用 `.numpy()` 轉換為純量

        # 合併正負樣本並生成標籤
        positive_similarities = np.array(positive_similarities).flatten()
        negative_similarities = np.array(negative_similarities).flatten()
        similarities = np.concatenate([positive_similarities, negative_similarities])  # 合併相似度
        labels = np.concatenate(
            [np.ones(len(positive_similarities)), np.zeros(len(negative_similarities))]  # 生成標籤
        )

        # 確保 similarities 和 labels 維度正確
        assert similarities.ndim == 1, f"similarities 維度應為 1，但得到 {similarities.ndim}"
        assert labels.ndim == 1, f"labels 維度應為 1，但得到 {labels.ndim}"

        # 計算 AUC
        auc = roc_auc_score(labels, similarities)
        auc_scores[target] = auc
        print(f"AUC: {auc:.4f}")


    return auc_scores

if __name__ == "__main__":
    # 獲取所有目標類別（除背景噪音外）
    wav_dir = "/datas/store162/syt/GE2E/DB/speech_commands"  # 指定數據集路徑
    all_targets = [
        d.name for d in Path(wav_dir).iterdir()
        if d.is_dir() and d.name not in ["_background_noise_"]
    ]

    print(f"Target List: {all_targets}")

    # 加載 GSC 測試集
    dataloader = GoogleCommandsDataloader(
        batch_size=2048,
        wav_dir=wav_dir,
        target_list=all_targets  # 使用自動生成的目標列表
    )

    print('DataLoader loaded')
    # 加載模型
    conformer_encoder = load_conformer_model(
        weights_path="./checkpoints/epoch_1_weights.h5",
        input_dim=80
    )

    # 評估
    auc_scores = evaluate_gsc(conformer_encoder, dataloader)
    print("AUC Scores:")
    for keyword, auc in auc_scores.items():
        print(f"{keyword}: {auc:.4f}")
