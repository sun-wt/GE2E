import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from conformer.conformer.encoder import ConformerEncoder
from collections import defaultdict
import torch.optim as optim
from criterion.GE2E_loss import GE2ELoss
import torch.nn as nn
import numpy as np
import librosa
import argparse
from datetime import datetime
import random
from torch.cuda.amp import autocast, GradScaler  # 引入混合精度訓練工具

# 設定設備
print(f"PyTorch 版本: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 特徵提取函數
def extract_features(wav_path, fs=16000, input_dim=40):
    """
    提取梅爾頻譜特徵並進行標準化
    """
    try:
        data, _ = librosa.load(wav_path, sr=fs)

        # 去除靜音部分
        data, _ = librosa.effects.trim(data, top_db=20)

        # 如果音頻不足 1 秒，進行零填充
        if len(data) < fs:
            data = np.pad(data, (0, fs - len(data)), mode='constant')
        else:
            data = data[:fs]

        # 提取梅爾頻譜特徵
        mel_spec = librosa.feature.melspectrogram(
            y=data,
            sr=fs,
            n_fft=1024,
            hop_length=256,
            n_mels=input_dim,
            fmax=fs // 2
        )

        # 轉換為對數梅爾頻譜
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 使用 Min-Max 標準化
        mel_spec_db_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-9)

        return mel_spec_db_normalized.T  # [time, n_mels]
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None


# Dataset 定義
class DynamicTrainDataset(Dataset):
    """
    動態從 MSWC_MIN_10.pkl 提取 wav 文件並轉換為特徵。
    """
    def __init__(self, pkl_path, batch_size=8, samples_per_label=10, input_dim=80, fs=16000, virtual_length=10000):
        # 載入數據（DataFrame 格式）
        self.data = pd.read_pickle(pkl_path)  # 確保 .pkl 文件為 pandas.DataFrame
        if not {'wav', 'WORD'}.issubset(self.data.columns):
            raise ValueError("The DataFrame must contain 'wav' and 'WORD' columns.")
        
        self.batch_size = batch_size
        self.samples_per_label = samples_per_label
        self.input_dim = input_dim
        self.fs = fs
        self.virtual_length = virtual_length

        # 使用 groupby 快速分組生成字典
        print("Grouping data by WORD...")
        self.word_to_samples = self.data.groupby('WORD')['wav'].apply(list).to_dict()

    def __len__(self):
        # 返回虛擬長度
        return self.virtual_length

    def __getitem__(self, idx):
        # 將單字列表從字典的 keys 中提取出來
        valid_words = list(self.word_to_samples.keys())

        # 隨機選擇 batch_size 個單字
        selected_words = random.sample(valid_words, self.batch_size)
        print(f"Selected words: {selected_words}")

        batch_features = []
        batch_labels = []

        for label_idx, word in enumerate(selected_words):
            # 從該單字中隨機選取 samples_per_label 個樣本
            selected_samples = random.sample(self.word_to_samples[word], self.samples_per_label)

            for wav_path in selected_samples:
                if not os.path.exists(wav_path):
                    print(f"[Warning] WAV 文件不存在: {wav_path}")
                    continue

                # 提取特徵
                features = extract_features(wav_path, fs=self.fs, input_dim=self.input_dim)
                if features is not None:
                    batch_features.append(torch.tensor(features, dtype=torch.float32))
                    batch_labels.append(label_idx)  # 每個單字對應一個整數標籤

        # 填充特徵到相同長度
        pad_features = pad_sequence(batch_features, batch_first=True)  # [batch_size * samples_per_label, max_time, input_dim]
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)  # [batch_size * samples_per_label]

        return pad_features, batch_labels

# 批次處理函數
def collate_fn(batch):
    """
    將特徵和標籤填充成固定形狀並轉換為張量。
    """
    features, labels = zip(*batch)  # features: tuple of [batch_size * samples_per_label, time, input_dim]
    features = torch.cat(features, dim=0)  # [total_samples, time, input_dim]
    labels = torch.cat(labels, dim=0)  # [total_samples]
    return features, labels

# 訓練函數
def train_model(model, train_dataloader, optimizer, loss_fn, device, epochs, save_checkpoint_dir):
    scaler = GradScaler()  # 初始化 GradScaler
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0.0
        all_auc = []

        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            input_lengths = torch.tensor([inputs.shape[1]] * inputs.shape[0], dtype=torch.long, device=device)

            # 混合精度前向傳播
            with autocast():
                emb, _ = model(inputs, input_lengths)
                loss, auc = loss_fn(emb, labels)  # 返回 Loss 和 AUC

            # 混合精度反向傳播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            all_auc.append(auc)  # 收集每個 batch 的 AUC

            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}, AUC: {auc:.4f}")

        avg_loss = total_loss / len(train_dataloader)
        avg_auc = sum(all_auc) / len(all_auc)  # 計算平均 AUC
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}, Average AUC: {avg_auc:.4f}")

        checkpoint_path = os.path.join(save_checkpoint_dir, f"epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[Info] 模型檢查點已保存至 {checkpoint_path}")


def main(args):

    # 加載數據集
    dataset = DynamicTrainDataset(
        pkl_path=args.pkl_path,
        batch_size=args.batch_size,
        samples_per_label=args.samples_per_label,
        input_dim=args.input_dim,
        virtual_length=args.virtual_length
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    # 初始化模型
    model = ConformerEncoder(
        input_dim=args.input_dim,
        encoder_dim=args.encoder_dim,
        num_layers=args.num_encoder_layers,
        num_attention_heads=args.num_attention_heads
    ).to(device)

    # 打印模型結構以確認最終線性層名稱
    print(model)

    # 加載檢查點
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint_name)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"[Info] 初始模型權重已從 {checkpoint_path} 加載。")
    else:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[Info] 初始模型權重已保存至 {checkpoint_path}")
        
    if args.use_alpha_beta:
        print("[Info] 使用 alpha 和 beta 參數。")
        
    # 設置損失函數和優化器
    loss_fn = GE2ELoss(device=device, use_alpha_beta=args.use_alpha_beta)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 創建檢查點保存目錄
    os.makedirs(args.save_dir, exist_ok=True)

    # 訓練模型
    train_model(model, dataloader, optimizer, loss_fn, device, args.epochs, args.save_dir)
    
def test(args):

    # 加載數據集
    dataset = DynamicTrainDataset(
        pkl_path=args.pkl_path,
        batch_size=args.batch_size,
        samples_per_label=args.samples_per_label,
        input_dim=args.input_dim,
        virtual_length=args.virtual_length
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    # 初始化模型
    model = ConformerEncoder(
        input_dim=args.input_dim,
        encoder_dim=args.encoder_dim,
        num_layers=args.num_encoder_layers,
        num_attention_heads=args.num_attention_heads
    ).to(device)

    # 打印模型結構以確認最終線性層名稱
    print(model)

    # 加載檢查點
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint_name)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"[Info] 初始模型權重已從 {checkpoint_path} 加載。")
    else:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[Info] 初始模型權重已保存至 {checkpoint_path}")
        
    # 設置損失函數和優化器
    loss_fn = GE2ELoss(device=device, use_alpha_beta=args.use_alpha_beta)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # **測試一個 batch**
    print("\n[Testing a single batch]")
    for step, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)  # [batch_size * samples_per_label, time, input_dim]
        labels = labels.to(device)

        # 計算每個序列的有效長度
        input_lengths = torch.tensor([inputs.shape[1]] * inputs.shape[0], dtype=torch.long, device=device)
        
        # 逐條打印每個 input
        print(f"Inputs content:")
        for i, ii in enumerate(inputs):
            print(f"Input {i + 1}: {ii.cpu().detach().numpy()}")

        # forward 傳遞，獲取輸出和有效長度
        outputs, output_lengths = model(inputs, input_lengths)

        # 打印中間結果
        print(f"Inputs shape: {inputs.shape}")
        print(f"Input lengths: {input_lengths}")
        print(f"Outputs shape: {outputs.shape}")
        print(f"Output lengths: {output_lengths}")

        # **進一步計算 embedding**
        embeddings = outputs.mean(dim=1)  # 平均池化作為示例
        print(f"Embeddings shape: {embeddings.shape}")

        # 逐條打印每個 embedding
        print(f"Embeddings content:")
        for i, embedding in enumerate(embeddings):
            print(f"Embedding {i + 1}: {embedding.cpu().detach().numpy()}")

        # # 如果需要計算損失
        loss = loss_fn(embeddings, labels)
        print(f"Loss: {loss.item()}")

        # 只測試一個 batch，停止迴圈
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with dynamic feature extraction")
    parser.add_argument('--pkl_path', type=str, required=True, help='Path to the MSWC_MIN_10.pkl file')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default='e0.pt', help='Checkpoint file name')  # 加入此行
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of words per batch')
    parser.add_argument('--samples_per_label', type=int, default=10, help='Number of samples per word')
    parser.add_argument('--input_dim', type=int, default=80, help='Feature dimension')
    parser.add_argument('--virtual_length', type=int, default=10000, help='Virtual length for DataLoader')
    parser.add_argument('--encoder_dim', type=int, default=128, help='Conformer encoder dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=4, help='Number of Conformer encoder layers')
    parser.add_argument('--num_attention_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--use_alpha_beta', action='store_true', help='Use alpha and beta in GE2E loss')

    args = parser.parse_args()
    main(args)
    # test(args)



