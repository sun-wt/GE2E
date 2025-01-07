import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
import matplotlib.pyplot as plt

import pickle
from dataset.google_v1 import GoogleCommandsDataloader

import sys
sys.path.append('/datas/store162/syt/GE2E/conformer')

from conformer.conformer.model import Conformer

import argparse

# 設置隨機種子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 計算 DET 曲線下的 AUC
def calculate_det_auc(fpr, fnr):
    """計算 DET 曲線（FRR-FAR 曲線）下的面積。"""
    return auc(fpr, fnr)

# 評估函數
def evaluate_gsc(model, enroll_data, test_data, num_enrollment=10):
    model.eval()  # 設置模型為評估模式
    auc_scores = {}
    det_curves = {}

    for target in enroll_data.keys():
        print(f"\nEvaluating keyword: {target}")

        # 加載 enroll 和 test 數據
        enroll_wavs = enroll_data[target]
        test_wavs = test_data[target]

        # 計算 enroll embeddings
        enroll_embeddings = []
        for enroll_feature in enroll_wavs:
            enroll_audio = torch.tensor(enroll_feature).unsqueeze(0).to(device)
            with torch.no_grad():
                enroll_embedding, _ = model(enroll_audio, torch.tensor([enroll_audio.shape[1]]).to(device))
            enroll_embeddings.append(enroll_embedding.mean(dim=1))  # 平均時間維度

        centroid = torch.stack(enroll_embeddings).mean(dim=0)  # 計算中心點

        # 測試數據 (正樣本)
        positive_similarities = []
        for test_feature in test_wavs:
            test_audio = torch.tensor(test_feature).unsqueeze(0).to(device)
            with torch.no_grad():
                test_embedding, _ = model(test_audio, torch.tensor([test_audio.shape[1]]).to(device))
            sim = torch.cosine_similarity(test_embedding.mean(dim=1), centroid)
            positive_similarities.append(sim.cpu().numpy())

        # 測試數據 (負樣本)
        negative_similarities = []
        for other_target in enroll_data.keys():
            if other_target != target:
                other_test_wavs = test_data[other_target]
                for other_test_feature in other_test_wavs:
                    other_test_audio = torch.tensor(other_test_feature).unsqueeze(0).to(device)
                    with torch.no_grad():
                        other_test_embedding, _ = model(other_test_audio, torch.tensor([other_test_audio.shape[1]]).to(device))
                    neg_sim = torch.cosine_similarity(other_test_embedding.mean(dim=1), centroid)
                    negative_similarities.append(neg_sim.cpu().numpy())

        # 合併正負樣本並生成標籤
        positive_similarities = np.array(positive_similarities).flatten()
        negative_similarities = np.array(negative_similarities).flatten()
        similarities = np.concatenate([positive_similarities, negative_similarities])
        labels = np.concatenate(
            [np.ones(len(positive_similarities)), np.zeros(len(negative_similarities))]
        )

        # 打印正負樣本數量
        print(f"Positive samples: {len(positive_similarities)}, Negative samples: {len(negative_similarities)}")

        # 確保 similarities 和 labels 維度正確
        assert similarities.ndim == 1, f"similarities 維度應為 1，但得到 {similarities.ndim}"
        assert labels.ndim == 1, f"labels 維度應為 1，但得到 {labels.ndim}"

        # 計算 DET 曲線
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        fnr = 1 - tpr  # 計算 False Negative Rate (FRR)
        det_auc = calculate_det_auc(fpr, fnr)
        auc_scores[target] = det_auc
        det_curves[target] = (fpr, fnr)

        print(f"AUC for {target}: {det_auc:.4f}")

    return auc_scores, det_curves

# 繪製 DET 曲線
def plot_det_curves(det_curves, save_path='det_curve.png'):
    plt.figure(figsize=(10, 8))
    for target, (fpr, fnr) in det_curves.items():
        plt.plot(fpr, fnr, label=f"{target}")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("False Negative Rate (FNR)")
    plt.title("DET Curve")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    print(f"DET 曲線已保存至 {save_path}")
    plt.close()


# 主程序
def main(args):
    # 加載 enroll 和 test 數據
    print(f"Loading enroll and test data.")
    with open(args.enroll_path, 'rb') as f:
        enroll_data = pickle.load(f)
    with open(args.test_path, 'rb') as f:
        test_data = pickle.load(f)

    print(f"Loaded enroll and test data.")

    # 初始化 Conformer 模型
    model = Conformer(
        num_classes=args.num_classes,          # 根據訓練時的設定
        input_dim=args.input_dim,            # 音頻特徵維度，例如 Mel 頻譜的頻帶數
        encoder_dim=args.encoder_dim,         # 編碼器維度
        num_encoder_layers=args.num_encoder_layers,    # 編碼器層數
        num_attention_heads=args.num_attention_heads    # 注意力頭數
    ).to(device)

    # 加載檢查點
    if args.checkpoint_path:
        if os.path.isfile(args.checkpoint_path):
            state_dict = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"[Info] 模型權重已從 {args.checkpoint_path} 加載。")
        else:
            print(f"[Error] 指定的檢查點路徑 {args.checkpoint_path} 不存在。")
            return
    else:
        print("[Error] 未指定檢查點路徑。使用 --checkpoint_path 參數指定要加載的檢查點。")
        return

    # 評估模型
    auc_scores, det_curves = evaluate_gsc(model, enroll_data, test_data)

    # 打印 AUC 分數
    print("\nAUC Scores:")
    for keyword, auc in auc_scores.items():
        print(f"{keyword}: {auc:.4f}")

    # 繪製 DET 曲線並保存
    plot_save_path = os.path.join(os.path.dirname(args.checkpoint_path), 'det_curve.png')
    plot_det_curves(det_curves, save_path=plot_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Conformer with GE2ELoss")

    parser.add_argument('--enroll_path', type=str, required=True, help='Path to the enroll pickle file')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test pickle file')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint (.pt) file')
    parser.add_argument('--num_classes', type=int, default=35, help='Number of classes (should match training)')
    parser.add_argument('--input_dim', type=int, default=80, help='Input feature dimension (should match training)')
    parser.add_argument('--encoder_dim', type=int, default=128, help='Encoder dimension (should match training)')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of encoder layers (should match training)')
    parser.add_argument('--num_attention_heads', type=int, default=4, help='Number of attention heads (should match training)')

    args = parser.parse_args()
    
    main(args)
