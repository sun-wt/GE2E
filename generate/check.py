# import os
# import pickle
# import numpy as np
# import torch

# def check_pkl_file(pkl_path):
#     """
#     檢查 .pkl 文件的 Mel 特徵和標籤統計信息。
#     Args:
#         pkl_path: .pkl 文件路徑
#     """
#     if not os.path.exists(pkl_path):
#         raise FileNotFoundError(f"文件不存在: {pkl_path}")

#     print(f"載入 {pkl_path}...")
#     with open(pkl_path, 'rb') as f:
#         data = pickle.load(f)

#     print(f"數據加載完成，共 {len(data)} 條樣本。")

#     # 分離特徵和標籤
#     features = [item[0] for item in data]
#     labels = [item[1] for item in data]

#     # 打印標籤信息
#     unique_labels = set(labels)
#     print(f"總共有 {len(unique_labels)} 個唯一標籤。")
#     print(f"前 10 個標籤: {list(unique_labels)[:10]}")

#     # 轉換為 numpy 數組檢查特徵統計信息
#     features_flat = np.concatenate([f.flatten() for f in features], axis=0)

#     print("Mel 特徵統計信息:")
#     print(f"  最小值: {features_flat.min():.4f}")
#     print(f"  最大值: {features_flat.max():.4f}")
#     print(f"  平均值: {features_flat.mean():.4f}")
#     print(f"  標準差: {features_flat.std():.4f}")

#     # 檢查是否有無效數據
#     if np.isnan(features_flat).any():
#         print("警告: Mel 特徵中包含 NaN 值！")
#     if np.isinf(features_flat).any():
#         print("警告: Mel 特徵中包含 Inf 值！")

#     # 打印樣本大小信息
#     feature_shapes = [f.shape for f in features]
#     print(f"樣本的 Mel 特徵維度統計: {set(feature_shapes)}")

#     # 檢查每個標籤的樣本數量
#     label_counts = {}
#     for label in labels:
#         label_counts[label] = label_counts.get(label, 0) + 1
#     print(f"標籤數量分佈 (前 10 個): {dict(list(label_counts.items())[:10])}")


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="檢查 .pkl 文件的 Mel 特徵和標籤信息")
#     parser.add_argument('--pkl_path', type=str, required=True, help='要檢查的 .pkl 文件路徑')

#     args = parser.parse_args()

#     check_pkl_file(args.pkl_path)


import pickle
from collections import Counter

def check_pkl_samples(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 統計每個單字的樣本數
    labels = [item[1] for item in data]
    label_counts = Counter(labels)

    print("[Word Sample Counts]")
    for word, count in label_counts.items():
        print(f"  Word '{word}': {count} samples")

if __name__ == "__main__":
    check_pkl_samples("generate/train_fixed_1000.pkl")
