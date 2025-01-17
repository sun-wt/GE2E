import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import librosa
import argparse
from pathlib import Path
from tqdm import tqdm

from espnet2.bin.asr_inference import Speech2Text  # 這裡使用 ESPnet2 的 ASR 模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_audio(wav_path, fs=16000):
    data, _ = librosa.load(wav_path, sr=fs)
    data, _ = librosa.effects.trim(data, top_db=20)
    if len(data) < fs:
        data = np.pad(data, (0, fs - len(data)), mode='constant')
    else:
        data = data[:fs]
    return data

def calculate_det_auc(fpr, fnr):
    return auc(fpr, fnr)

def evaluate_gsc_espnet(asr_model, enroll_data, test_data):
    asr_model.eval()
    auc_scores = {}
    det_curves = {}

    # 遍歷所有關鍵詞進行評估
    for target in tqdm(enroll_data.keys(), desc="Evaluating keywords"):
        print(f"\nEvaluating keyword: {target}")

        enroll_files = enroll_data[target]
        test_files = test_data[target]

        # Step 1: 計算中心點（Centroid）
        enroll_embeddings = []
        for wav_path in enroll_files:
            data = load_audio(wav_path)
            waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)  # shape: [1, T]
            lengths = torch.tensor([waveform.shape[1]], device=device)

            with torch.no_grad():
                feats, _ = asr_model.encode(waveform, lengths)  # 假設 asr_model 有 encode 方法
            emb = feats.mean(dim=1)  # 平均池化獲取嵌入
            enroll_embeddings.append(emb)

        # 計算中心點向量
        centroid = torch.stack(enroll_embeddings, dim=0).mean(dim=0).squeeze(0)  # 確保形狀為 [embedding_dim]

        # Step 2: 計算正樣本相似度
        positive_similarities = []
        for wav_path in tqdm(test_files, desc=f"Positive for '{target}'", leave=False):
            data = load_audio(wav_path)
            waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
            lengths = torch.tensor([waveform.shape[1]], device=device)

            with torch.no_grad():
                feats, _ = asr_model.encode(waveform, lengths)
            emb = feats.mean(dim=1).squeeze(0)  # 確保形狀為 [embedding_dim]

            # 確保 sim 的輸出為標量
            sim = torch.cosine_similarity(emb.unsqueeze(0), centroid.unsqueeze(0), dim=1)  # 輸出為 1D 張量
            positive_similarities.append(sim.item())  # 提取標量值

        # Step 3: 計算負樣本相似度
        negative_similarities = []
        for other_target, other_files in tqdm(test_data.items(), desc=f"Negative keywords for '{target}'"):
            if other_target == target:
                continue

            for wav_path in tqdm(other_files, desc=f"Samples from '{other_target}'", leave=False):
                data = load_audio(wav_path)
                waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
                lengths = torch.tensor([waveform.shape[1]], device=device)

                with torch.no_grad():
                    feats, _ = asr_model.encode(waveform, lengths)
                emb = feats.mean(dim=1).squeeze(0)  # 確保形狀為 [embedding_dim]

                # 確保 sim 的輸出為標量
                sim = torch.cosine_similarity(emb.unsqueeze(0), centroid.unsqueeze(0), dim=1)  # 輸出為 1D 張量
                negative_similarities.append(sim.item())  # 提取標量值

        # Step 4: 計算 AUC
        positive_similarities = np.array(positive_similarities)
        negative_similarities = np.array(negative_similarities)

        # 合併正樣本和負樣本的相似度
        similarities = np.concatenate([positive_similarities, negative_similarities])
        labels = np.concatenate([
            np.ones(len(positive_similarities)),  # 正樣本標籤為 1
            np.zeros(len(negative_similarities))  # 負樣本標籤為 0
        ])

        print(f"Positive samples: {len(positive_similarities)}, Negative samples: {len(negative_similarities)}")
        assert similarities.ndim == 1
        assert labels.ndim == 1

        # 計算 ROC 曲線和 DET AUC
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        fnr = 1 - tpr
        det_auc = calculate_det_auc(fpr, fnr)
        auc_scores[target] = det_auc
        det_curves[target] = (fpr, fnr)

        print(f"AUC for {target}: {det_auc:.4f}")

    return auc_scores, det_curves

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

def main(args):
    # 使用 pathlib 獲取所有關鍵詞
    wav_dir = Path(args.data_dir)
    all_targets = [
        d.name for d in wav_dir.iterdir()
        if d.is_dir() and d.name not in ["_background_noise_"]
    ]
    print(f"Found keywords: {all_targets}")

    # 指定要評估的關鍵詞或使用全部
    if args.keywords:
        user_keywords = [k.strip() for k in args.keywords.split(',')]
        eval_keywords = [k for k in all_targets if k in user_keywords]
        print(f"Evaluating specified keywords: {eval_keywords}")
    else:
        eval_keywords = all_targets
        print("No specific keywords provided; evaluating all found keywords.")

    # 構建 enrollment 和 test 數據路徑字典
    enroll_data = {kw: [] for kw in eval_keywords}
    test_data = {kw: [] for kw in eval_keywords}

    for keyword in eval_keywords:
        keyword_dir = wav_dir / keyword
        all_files = sorted([str(f) for f in keyword_dir.glob("*.wav")])
        enroll_data[keyword] = all_files[:10]  # 前10個作為 enrollment
        test_data[keyword] = all_files[10:]    # 其餘作為 test

    # 初始化 ESPnet ASR 模型
    print("Loading ESPnet ASR model...")
    speech2text = Speech2Text.from_pretrained("Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best")
    asr_model = speech2text.asr_model.to(device)

    # 評估模型
    auc_scores, det_curves = evaluate_gsc_espnet(asr_model, enroll_data, test_data)

    print("\nAUC Scores:")
    for keyword, auc in auc_scores.items():
        print(f"{keyword}: {auc:.4f}")

    # 繪製 DET 曲線並保存
    plot_save_path = os.path.join(os.getcwd(), 'det_curve.png')
    plot_det_curves(det_curves, save_path=plot_save_path)

if __name__ == "__main__":
    from espnet2.bin.asr_inference import Speech2Text
    parser = argparse.ArgumentParser(description="Evaluate ESPnet ASR model with GE2ELoss-like methodology")
    parser.add_argument('--data_dir', type=str, default='/datas/store162/syt/GE2E/DB/google_speech_commands', help='Root directory of speech commands dataset')
    parser.add_argument('--keywords', type=str, required=False, help='Comma-separated list of keywords to evaluate')
    parser.add_argument('--input_dim', type=int, default=40, help='Input feature dimension')  # 这里主要针对特征提取，如果使用ESPnet原始波形可能不需要
    parser.add_argument('--encoder_dim', type=int, default=64, help='Encoder dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=5, help='Number of encoder layers')
    parser.add_argument('--num_attention_heads', type=int, default=2, help='Number of attention heads')

    args = parser.parse_args()
    main(args)
