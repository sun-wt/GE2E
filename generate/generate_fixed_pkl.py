import os
import numpy as np
import pandas as pd
import librosa
import pickle
from tqdm import tqdm
import argparse
import random
from multiprocessing import Pool, cpu_count
from collections import defaultdict

def extract_features(wav_path, fs=16000, input_dim=80):
    """
    提取梅爾頻譜特徵並進行標準化
    """
    if not isinstance(wav_path, str):
        raise ValueError(f"Invalid file: {wav_path}")

    try:
        data, _ = librosa.load(wav_path, sr=fs)
    except Exception as e:
        print(f"Error loading {wav_path}: {e}")
        return None

    if len(data) < fs:
        data = np.pad(data, (0, fs - len(data)), mode='constant')
    else:
        data = data[:fs]

    # 計算音頻能量
    energy = np.sum(data ** 2) / len(data)
    if energy < 1e-6:
        data = data * (1e-6 / energy) ** 0.5

    mel_spec = librosa.feature.melspectrogram(y=data, sr=fs, n_mels=input_dim, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=100)

    # 標準化
    mel_spec_db_normalized = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-9)

    return mel_spec_db_normalized.T  # [time, n_mels]

def process_sample(row):
    wav_path = row['wav']
    label = row['WORD']
    if not os.path.exists(wav_path):
        print(f"WAV 文件不存在: {wav_path}")
        return None
    features = extract_features(wav_path)
    if features is not None:
        return (features, label)
    else:
        print(f"無法提取特徵: {wav_path}")
        return None

def generate_fixed_pkl(input_pkl, output_pkl, batch_size=8, samples_per_label=10, num_batches=200):
    """
    從輸入的 pkl 文件中隨機選擇 8 個單字，每個單字選取 10 個樣本，重複 200 次，
    提取特徵，並保存到輸出 pkl 文件。
    """
    # 載入原始 pkl 文件
    print(f"載入 {input_pkl}...")
    df = pd.read_pickle(input_pkl)
    print(f"載入完成，形狀為: {df.shape}")

    # 獲取所有唯一的標籤且有至少 samples_per_label 個樣本
    print("篩選具有足夠樣本的標籤...")
    label_counts = df['WORD'].value_counts()
    eligible_labels = label_counts[label_counts >= samples_per_label].index.tolist()
    num_eligible_labels = len(eligible_labels)
    print(f"共有 {num_eligible_labels} 個標籤具有至少 {samples_per_label} 筆樣本。")

    if num_eligible_labels == 0:
        raise ValueError("沒有任何標籤具有足夠的樣本數量。請檢查數據。")

    fixed_data = []
    used_labels = set()  # 記錄已選過的單字
    label_history = defaultdict(list)  # 保存每批次的選擇歷史

    # 設置隨機種子
    random.seed(42)

    # 使用 pandas groupby 高效分組
    groups = df.groupby('WORD')

    for batch_num in tqdm(range(num_batches), desc="生成批次"):
        # 從未選過的標籤中隨機選擇
        available_labels = [label for label in eligible_labels if label not in used_labels]
        if len(available_labels) < batch_size:
            print(f"[Warning] 可用標籤不足，剩餘 {len(available_labels)} 個，重新開始選擇。")
            used_labels.clear()
            available_labels = [label for label in eligible_labels if label not in used_labels]

        selected_labels = random.sample(available_labels, batch_size)
        used_labels.update(selected_labels)

        # 記錄選擇的標籤
        label_history[batch_num + 1] = selected_labels

        # 對每個標籤，隨機選取 samples_per_label 個樣本
        for label in selected_labels:
            group = groups.get_group(label)
            sampled_df = group.sample(n=samples_per_label, replace=False, random_state=random.randint(0, 10000))
            fixed_data.extend(sampled_df[['wav', 'WORD']].to_dict('records'))

    # 將所有選取的樣本進行特徵提取
    print("開始提取特徵...")
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_sample, fixed_data), total=len(fixed_data), desc="提取特徵"))

    # 移除 None 的結果
    fixed_data_features = [item for item in results if item is not None]

    total_expected = batch_size * samples_per_label * num_batches  # 8 * 10 * 200 = 16000
    print(f"收集到的固定樣本數量: {len(fixed_data_features)} / 預期: {total_expected}")

    if len(fixed_data_features) < total_expected:
        print(f"警告: 收集的樣本數量 ({len(fixed_data_features)}) 少於預期 ({total_expected})。")

    # 隨機打亂固定資料
    np.random.seed(42)
    np.random.shuffle(fixed_data_features)

    # 選取前 total_expected 筆樣本
    fixed_data_features = fixed_data_features[:total_expected]

    # 保存單字選擇記錄
    label_history_path = output_pkl.replace('.pkl', '_label_history.txt')
    print(f"保存單字選擇記錄到 {label_history_path}...")
    with open(label_history_path, 'w') as f:
        for batch_num, labels in label_history.items():
            f.write(f"Batch {batch_num}: {', '.join(labels)}\n")

    # 保存到 pkl 文件
    print(f"保存固定資料到 {output_pkl}...")
    with open(output_pkl, 'wb') as f:
        pickle.dump(fixed_data_features, f)
    print(f"已將固定資料保存至 {output_pkl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="從 MSWC_MIN_10.pkl 生成 train_fixed.pkl")
    parser.add_argument('--input_pkl', type=str, required=True, help='輸入的 MSWC_MIN_10.pkl 文件路徑')
    parser.add_argument('--output_pkl', type=str, required=True, help='輸出的 train_fixed.pkl 文件路徑')
    parser.add_argument('--batch_size', type=int, default=8, help='每個批次的標籤數量')
    parser.add_argument('--samples_per_label', type=int, default=10, help='每個標籤每批次的樣本數量')
    parser.add_argument('--num_batches', type=int, default=200, help='要生成的批次數量')

    args = parser.parse_args()

    generate_fixed_pkl(
        input_pkl=args.input_pkl,
        output_pkl=args.output_pkl,
        batch_size=args.batch_size,
        samples_per_label=args.samples_per_label,
        num_batches=args.num_batches
    )
