import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

import sys
sys.path.append('/datas/store162/syt/GE2E/conformer')

from conformer.conformer.model import Conformer

from dataset.google import GoogleCommandsDataloader
from pathlib import Path

# 設置隨機種子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 創建 Conformer 模型
model = Conformer(
    num_classes=35,          # 假設 10 個分類
    input_dim=80,            # 音頻特徵維度，例如 Mel 頻譜的頻帶數
    encoder_dim=128,         # 編碼器維度
    num_encoder_layers=6,    # 編碼器層數
    num_attention_heads=4    # 注意力頭數
).to(device)

# 評估函數
def evaluate_gsc(model, dataloader, num_enrollment=10):
    model.eval()  # 設置模型為評估模式
    auc_scores = {}
    target_list = dataloader.target_list

    for target in target_list:
        print(f"Evaluating keyword: {target}")

        # 選取 enroll 和 test 數據
        enroll_indices = dataloader.data[dataloader.data['keyword'] == target].index[:num_enrollment]
        test_indices = dataloader.data[dataloader.data['keyword'] == target].index[num_enrollment:]
        
        # 計算 enroll embeddings
        enroll_embeddings = []
        for i in enroll_indices:
            wav = dataloader.data.loc[i, 'wav']
            enroll_audio = dataloader._load_wav(wav)
            enroll_audio = torch.tensor(enroll_audio).unsqueeze(0).to(device)
            with torch.no_grad():
                enroll_embedding, _ = model(enroll_audio, torch.tensor([enroll_audio.shape[1]]).to(device))
            enroll_embeddings.append(enroll_embedding.mean(dim=1))  # 平均時間維度

        centroid = torch.stack(enroll_embeddings).mean(dim=0)  # 計算中心點

        # 測試數據 (正樣本)
        positive_similarities = []
        for i in test_indices:
            wav = dataloader.data.loc[i, 'wav']
            test_audio = dataloader._load_wav(wav)
            test_audio = torch.tensor(test_audio).unsqueeze(0).to(device)
            with torch.no_grad():
                test_embedding, _ = model(test_audio, torch.tensor([test_audio.shape[1]]).to(device))
            sim = torch.cosine_similarity(test_embedding.mean(dim=1), centroid)
            positive_similarities.append(sim.cpu().numpy())

        # 測試數據 (負樣本)
        negative_similarities = []
        for other_target in target_list:
            if other_target != target:
                other_test_indices = dataloader.data[dataloader.data['keyword'] == other_target].index[:1]
                for i in other_test_indices:
                    wav = dataloader.data.loc[i, 'wav']
                    other_test_audio = dataloader._load_wav(wav)
                    other_test_audio = torch.tensor(other_test_audio).unsqueeze(0).to(device)
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

        # 確保 similarities 和 labels 維度正確
        assert similarities.ndim == 1, f"similarities 維度應為 1，但得到 {similarities.ndim}"
        assert labels.ndim == 1, f"labels 維度應為 1，但得到 {labels.ndim}"

        # 計算 AUC
        auc = roc_auc_score(labels, similarities)
        auc_scores[target] = auc
        print(f"AUC for {target}: {auc:.4f}")

    return auc_scores


# 主程序
if __name__ == "__main__":
    # 獲取所有目標類別（除背景噪音外）
    wav_dir = "/datas/store162/syt/GE2E/DB/google_speech_commands"
    all_targets = [
        d.name for d in Path(wav_dir).iterdir()
        if d.is_dir() and d.name not in ["_background_noise_"]
    ]

    print(f"Target List: {all_targets}")

    # 加載 Google Commands 數據集
    dataloader = GoogleCommandsDataloader(
        batch_size=2048,
        wav_dir=wav_dir,
        target_list=all_targets
    )

    print('DataLoader loaded')

    # 評估模型
    auc_scores = evaluate_gsc(model, dataloader)
    print("AUC Scores:")
    for keyword, auc in auc_scores.items():
        print(f"{keyword}: {auc:.4f}")
