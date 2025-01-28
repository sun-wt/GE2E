import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle
import argparse
import librosa
from tqdm import tqdm
from pathlib import Path

sys.path.append('/datas/store162/syt/GE2E/conformer')
from conformer.conformer.model import ConformerEncoder
from tiny_conformer.conformer.encoder import ConformerEncoder as TinyConformerEncoder

# from lstm import ThreeLayerLSTM  # 若需要使用 LSTM 可取消註解

# 設置隨機種子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def extract_features(wav_path, fs=16000, input_dim=40):
    try:
        data, _ = librosa.load(wav_path, sr=fs)
        data, _ = librosa.effects.trim(data, top_db=20)
        if len(data) < fs:
            data = np.pad(data, (0, fs - len(data)), mode='constant')
        else:
            data = data[:fs]
        mel_spec = librosa.feature.melspectrogram(
            y=data,
            sr=fs,
            n_fft=1024,
            hop_length=256,
            n_mels=input_dim,
            fmax=fs // 2
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-9)
        return mel_spec_db_normalized.T  # [time, n_mels]
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

def load_enroll_and_test_q(data_root, keywords, input_dim=80, enroll_samples=10):
    """
    加載 enrollment 和 test 數據，適配高通資料夾結構：
    Qualcomm/
    ├── keyword1/
    │   ├── speaker1/
    │   │   ├── audio1.wav
    │   │   ├── audio2.wav
    │   ├── speaker2/
    │       ├── audio3.wav
    │       ├── audio4.wav
    ├── keyword2/
    │   ├── speaker3/
    │   ├── speaker4/
    """
    enroll_data = {}
    test_data = {}

    for keyword in keywords:
        keyword_dir = Path(data_root) / keyword
        if not keyword_dir.is_dir():
            print(f"Warning: Directory for keyword '{keyword}' does not exist.")
            continue

        # 遍歷 speaker 資料夾並收集所有音檔
        all_files = []
        for speaker_dir in keyword_dir.iterdir():
            if speaker_dir.is_dir():
                speaker_files = list(speaker_dir.glob("*.wav"))
                all_files.extend(speaker_files)

        if len(all_files) < enroll_samples:
            print(f"Warning: Not enough samples for keyword '{keyword}'. Found {len(all_files)} samples.")
            continue

        # 隨機打亂並分配 enrollment 和 test 數據
        # random.shuffle(all_files)
        enroll_files = all_files[:enroll_samples]
        test_files = all_files[enroll_samples:]
        print(f"Keyword: {keyword}, Enrollment: {len(enroll_files)}, Test: {len(test_files)}")

        # 提取特徵
        enroll_features = []
        for wav_path in enroll_files:
            feat = extract_features(str(wav_path), input_dim=input_dim)
            if feat is not None:
                enroll_features.append(feat)

        test_features = []
        for wav_path in test_files:
            feat = extract_features(str(wav_path), input_dim=input_dim)
            if feat is not None:
                test_features.append(feat)

        # 保存特徵到結果字典
        enroll_data[keyword] = enroll_features
        test_data[keyword] = test_features

        if enroll_features:
            print(f"Enrollment feature shape for {keyword}: {enroll_features[0].shape}")

    return enroll_data, test_data

def load_enroll_and_test(data_root, keywords, input_dim=80, enroll_samples=10):
    enroll_data = {}
    test_data = {}
    for keyword in keywords:
        keyword_dir = os.path.join(data_root, keyword)
        if not os.path.isdir(keyword_dir):
            print(f"Warning: Directory for keyword '{keyword}' does not exist.")
            continue
        all_files = [os.path.join(keyword_dir, f) for f in os.listdir(keyword_dir) if f.endswith('.wav')]
        if len(all_files) < enroll_samples:
            print(f"Warning: Not enough samples for keyword '{keyword}'.")
            continue

        enroll_files = all_files[:enroll_samples]
        test_files = all_files[enroll_samples:]
        print(f"Keyword: {keyword}, Enrollment: {len(enroll_files)}, Test: {len(test_files)}")

        enroll_features = []
        for wav_path in enroll_files:
            feat = extract_features(wav_path, input_dim=input_dim)
            if feat is not None:
                enroll_features.append(feat)
        test_features = []
        for wav_path in test_files:
            feat = extract_features(wav_path, input_dim=input_dim)
            if feat is not None:
                test_features.append(feat)

        enroll_data[keyword] = enroll_features
        test_data[keyword] = test_features
        if enroll_features:
            print(f"Enrollment feature shape for {keyword}: {enroll_features[0].shape}")
    return enroll_data, test_data

def calculate_det_auc(fpr, fnr):
    return auc(fpr, fnr)

def evaluate_gsc(model, enroll_data, test_data, num_enrollment=10):
    model.eval()
    auc_scores = {}
    det_curves = {}

    for target in enroll_data.keys():
        print(f"\nEvaluating keyword: {target}")

        enroll_wavs = enroll_data[target]
        test_wavs = test_data[target]

        enroll_embeddings = []
        for enroll_feature in enroll_wavs:
            enroll_audio = torch.tensor(enroll_feature, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                enroll_embedding, _ = model(enroll_audio, torch.tensor([enroll_audio.shape[1]], device=device))
            enroll_embeddings.append(enroll_embedding.mean(dim=1))

        centroid = torch.stack(enroll_embeddings, dim=0).mean(dim=0)

        positive_similarities = []
        for test_feature in tqdm(test_wavs, desc=f"Positive samples for '{target}'"):
            test_audio = torch.tensor(test_feature, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                test_embedding, _ = model(test_audio, torch.tensor([test_audio.shape[1]], device=device))
            sim = torch.cosine_similarity(test_embedding.mean(dim=1), centroid, dim=1)
            positive_similarities.append(sim.item())

        negative_similarities = []
        for other_target, other_test_wavs in tqdm(test_data.items(), desc=f"Negative keywords for '{target}'"):
            if other_target == target:
                continue
            for other_test_feature in other_test_wavs:
                # print(f"Testing against {other_target}")  # 如有需要可取消註解以打印詳細信息
                other_test_audio = torch.tensor(other_test_feature, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    other_test_embedding, _ = model(other_test_audio, torch.tensor([other_test_audio.shape[1]], device=device))
                neg_sim = torch.cosine_similarity(other_test_embedding.mean(dim=1), centroid, dim=1)
                negative_similarities.append(neg_sim.item())

        positive_similarities = np.array(positive_similarities)
        negative_similarities = np.array(negative_similarities)
        similarities = np.concatenate([positive_similarities, negative_similarities])
        labels = np.concatenate([
            np.ones(len(positive_similarities)),
            np.zeros(len(negative_similarities))
        ])

        print(f"Positive samples: {len(positive_similarities)}, Negative samples: {len(negative_similarities)}")
        assert similarities.ndim == 1
        assert labels.ndim == 1

        fpr, tpr, thresholds = roc_curve(labels, similarities)
        fnr = 1 - tpr
        det_auc = calculate_det_auc(fpr, fnr)
        auc_scores[target] = det_auc
        det_curves[target] = (fpr, fnr)
        print(f"AUC for {target}: {det_auc:.4f}")

    return auc_scores, det_curves

def plot_det_curves(det_curves, save_dir='results/det_curves', overall_save_path='overall_det_curve.png'):
    """繪製每個關鍵字的 DET 曲線並保存，最後生成綜合 DET 曲線"""
    os.makedirs(save_dir, exist_ok=True)

    # 單獨為每個關鍵字繪製 DET 曲線
    for target, (fpr, fnr) in det_curves.items():
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, fnr, label=f"{target}")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("False Negative Rate (FNR)")
        plt.title(f"DET Curve for '{target}'")
        plt.legend()
        plt.grid()
        target_save_path = os.path.join(save_dir, f"{target}_det_curve.png")
        plt.savefig(target_save_path)
        print(f"DET 曲線已保存至 {target_save_path}")
        plt.close()

    # 綜合所有關鍵字的 DET 曲線
    plt.figure(figsize=(12, 8))
    for target, (fpr, fnr) in det_curves.items():
        plt.plot(fpr, fnr, label=f"{target}")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("False Negative Rate (FNR)")
    plt.title("Overall DET Curve")
    plt.legend()
    plt.grid()
    plt.savefig(overall_save_path)
    print(f"綜合 DET 曲線已保存至 {overall_save_path}")
    plt.close()

def main(args):
    # 檢查是否存在 pickle 文件
    if os.path.exists(args.enroll_path) and os.path.exists(args.test_path):
        print("Loading data from pickle files...")
        with open(args.enroll_path, 'rb') as f:
            enroll_data = pickle.load(f)
        with open(args.test_path, 'rb') as f:
            test_data = pickle.load(f)
        print("Data loaded from pickle files.")
    else:
        # 使用 pathlib 獲取所有關鍵詞
        wav_dir = Path(args.data_dir)
        all_targets = [
            d.name for d in wav_dir.iterdir()
            if d.is_dir() and d.name not in ["_background_noise_"]
        ]
        print(f"Found keywords: {all_targets}")

        enroll_data, test_data = load_enroll_and_test(
            data_root=args.data_dir,
            keywords=all_targets,
            input_dim=args.input_dim,
            enroll_samples=10
        )
        print("Loaded enroll and test data.")

        # 保存數據為 pickle 文件
        with open(args.enroll_path, 'wb') as f:
            pickle.dump(enroll_data, f)
            print("Enrollment data saved to enroll.pkl")
        with open(args.test_path, 'wb') as f:
            pickle.dump(test_data, f)
            print("Test data saved to test.pkl")

    # 如果用戶指定關鍵詞，用來過濾 enrollment 數據進行評估
    if args.keywords:
        user_keywords = [k.strip() for k in args.keywords.split(',')]
        eval_keywords = [k for k in user_keywords if k in enroll_data]
        print(f"Evaluating specified keywords: {eval_keywords}")
    else:
        eval_keywords = list(enroll_data.keys())
        print("No specific keywords provided; evaluating all keywords.")

    # 初始化模型
    if args.model_type == "tiny":
        print("[Info] 使用 Tiny Conformer 模型")
        model = TinyConformerEncoder(
            input_dim=args.input_dim,
            encoder_dim=args.encoder_dim,
            num_layers=args.num_encoder_layers,
            num_attention_heads=args.num_attention_heads
        ).to(device)
    else:
        print("[Info] 使用普通 Conformer 模型")
        model = ConformerEncoder(
            input_dim=args.input_dim,
            encoder_dim=args.encoder_dim,
            num_layers=args.num_encoder_layers,
            num_attention_heads=args.num_attention_heads
        ).to(device)

    # 加載檢查點
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        state_dict = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"[Info] 模型權重已從 {args.checkpoint_path} 加載。")
    else:
        print(f"[Error] 檢查點路徑 {args.checkpoint_path} 不存在。")
        return

    # 過濾出評估關鍵詞的 enrollment 數據
    filtered_enroll_data = {k: enroll_data[k] for k in eval_keywords if k in enroll_data}

    # 評估模型，負樣本仍來自所有非目標關鍵詞的測試數據
    auc_scores, det_curves = evaluate_gsc(model, filtered_enroll_data, test_data)

    print("\nAUC Scores:")
    for keyword, auc in auc_scores.items():
        print(f"{keyword}: {auc:.4f}")
    
    det_curve_dir = os.path.join(os.path.dirname(args.results_dir), 'det_curves')
    overall_curve_path = os.path.join(os.path.dirname(args.results_dir), 'overall_det_curve.png')

    # 繪製並保存 DET 曲線
    plot_det_curves(det_curves, save_dir=det_curve_dir, overall_save_path=overall_curve_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Conformer with GE2ELoss")
    parser.add_argument('--data_dir', type=str, default='/datas/store162/syt/GE2E/DB/google_speech_commands', help='Root directory of speech commands dataset')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save evaluation results')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint (.pt) file')
    parser.add_argument('--enroll_path', type=str, default='DB/enroll.pkl', help='Path to the enrollment data pickle file')    
    parser.add_argument('--test_path', type=str, default='DB/test.pkl', help='Path to the enrollment data pickle file')    
    parser.add_argument('--keywords', type=str, required=False, help='Comma-separated list of keywords to evaluate')
    parser.add_argument('--model_type', type=str, required=True, choices=['normal', 'tiny'], help='Model type to use (conformer or tiny_conformer)')
    parser.add_argument('--input_dim', type=int, default=40, help='Input feature dimension')
    parser.add_argument('--encoder_dim', type=int, default=64, help='Encoder dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=5, help='Number of encoder layers')
    parser.add_argument('--num_attention_heads', type=int, default=2, help='Number of attention heads')

    args = parser.parse_args()
    main(args)
    
    
