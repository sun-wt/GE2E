import os
import torch
from torch.utils.data import DataLoader, Dataset
import pickle
from torch.nn.utils.rnn import pad_sequence
from conformer.conformer.model import Conformer
from collections import defaultdict
import torch.optim as optim
from criterion.GE2E_loss import GE2ELoss
import torch.nn as nn
import numpy as np
import argparse
from datetime import datetime
import random

# Dataset 定義
class FixedTrainDataset(Dataset):
    """
    每個 __getitem__ 返回一組 (features, labels)，每組包含 8 個單字，每個單字有 10 個樣本。
    """
    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.all_data = pickle.load(f)
        self.features = [torch.tensor(item[0], dtype=torch.float32) for item in self.all_data]
        self.labels = [item[1] for item in self.all_data]

        # 檢查數據是否包含 NaN 或 Inf
        for i, feature in enumerate(self.features):
            if torch.isnan(feature).any() or torch.isinf(feature).any():
                raise ValueError(f"Feature at index {i} contains NaN or Inf values.")

        # 將數據按單字分組
        self.word_to_samples = defaultdict(list)
        for feature, label in zip(self.features, self.labels):
            self.word_to_samples[label].append(feature)

        # 確保每個單字的樣本數量滿足條件
        for word, samples in self.word_to_samples.items():
            if len(samples) != 10:
                self.word_to_samples[word] = random.sample(samples, 10)

        # 建立單字到索引的映射
        self.unique_labels = sorted(list(self.word_to_samples.keys()))
        self.label_to_index = {label: idx for idx, label in enumerate(self.unique_labels)}

        # 按照 8 個單字為一組構建批次
        self.batches = []
        words = list(self.word_to_samples.keys())
        for i in range(0, len(words), 8):  # 每次選擇 8 個單字
            batch_words = words[i:i + 8]
            batch_features = []
            batch_labels = []
            for word in batch_words:
                batch_features.extend(self.word_to_samples[word])  # 每個單字 10 個樣本
                batch_labels.extend([self.label_to_index[word]] * 10)  # 每個單字 10 次對應的標籤索引
            self.batches.append((batch_features, batch_labels))  # 保存特徵和對應標籤

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch_features, batch_labels = self.batches[idx]
        batch_tensor = torch.stack(batch_features)  # [80, time, feature_dim]
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)  # [80]
        return batch_tensor, batch_labels

# 批次處理函數
def collate_fn(batch):
    """
    將特徵和標籤填充成固定形狀並轉換為張量。
    """
    features, labels = zip(*batch)  # features: tuple of [80, time, feature_dim], labels: tuple of [80]
    # 由於每個 batch 的 features 已經是固定大小（80, time, feature_dim），不需要填充
    features = torch.stack(features)  # [batch_size, 80, time, feature_dim]
    labels = torch.stack(labels)  # [batch_size, 80]
    return features, labels

def train_model(model, train_dataloader, optimizer, loss_fn, device, epochs, save_checkpoint_dir):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0.0

        for step, (inputs, labels) in enumerate(train_dataloader):
            # 移動到 GPU / device
            inputs = inputs.to(device)  # [batch_size, 80, time, feature_dim]
            labels = labels.to(device)  # [batch_size, 80]

            # 檢查 inputs 是否含有 NaN 或 Inf
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                raise ValueError(f"Inputs contain NaN or Inf values at step {step}")

            # Reshape inputs for the model
            # 假設 Conformer 模型的輸入形狀為 [batch_size * 80, time, feature_dim]
            batch_size, num_samples, time, feature_dim = inputs.shape
            inputs = inputs.view(batch_size * num_samples, time, feature_dim)  # [batch_size * 80, time, feature_dim]
            labels = labels.view(batch_size * num_samples)  # [batch_size * 80]

            # 建立 input_lengths
            input_lengths = torch.tensor([inputs.shape[1]] * inputs.shape[0], dtype=torch.long, device=device)

            # forward
            emb, _ = model(inputs, input_lengths)  # emb: [batch_size * 80, embedding_dim]

            # 檢查 embeddings 是否含有 NaN 或 Inf
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                raise ValueError(f"Embeddings contain NaN or Inf values at step {step}")

            # 打印嵌入的統計信息
            # if step % 10 == 0:
                # print(f"Embeddings stats: min={emb.min().item():.4f}, max={emb.max().item():.4f}, mean={emb.mean().item():.4f}, std={emb.std().item():.4f}")

            # 計算損失
            try:
                loss = loss_fn(emb, labels)  # GE2ELoss
            except Exception as e:
                print(f"Error in loss computation at step {step}: {e}")
                raise

            # 檢查 loss 是否為 NaN 或 Inf
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Labels in Batch: {labels.cpu().numpy()}")
                raise ValueError(f"Loss contains NaN or Inf values at step {step}")

            # backward
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
                # 打印部分標籤以確認
                sample_labels = labels.cpu().numpy()[:10]  # 打印前 10 個標籤
                # print(f"Sample Labels: {sample_labels}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

        # 保存檢查點
        checkpoint_path = os.path.join(save_checkpoint_dir, f"epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[Info] 模型檢查點已保存至 {checkpoint_path}")

def main(args):
    # 設定設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加載數據集
    dataset = FixedTrainDataset(args.pkl_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,  # 從參數設置批次大小
        shuffle=args.shuffle,
        collate_fn=collate_fn
    )

    # 設置 num_classes 為數據集中唯一標籤的數量
    num_classes = 35

    # 初始化模型
    model = Conformer(
        num_classes=num_classes, 
        input_dim=args.input_dim,
        encoder_dim=args.encoder_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_attention_heads=args.num_attention_heads,
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

    # 創建以當前時間命名的子資料夾
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_checkpoint_dir = os.path.join(checkpoint_dir, timestamp)
    os.makedirs(save_checkpoint_dir, exist_ok=True)
    print(f"[Info] 保存檢查點到 {save_checkpoint_dir}")

    # 確認模型參數的 requires_grad
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Warning: {name} 不需要梯度。")

    # 設置損失函數和優化器
    loss_fn = GE2ELoss(device=device)  # 維持 GE2ELoss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 訓練模型
    train_model(model, dataloader, optimizer, loss_fn, device, epochs=args.epochs, save_checkpoint_dir=save_checkpoint_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conformer with GE2ELoss")
    parser.add_argument('--pkl_path', type=str, required=True, help='Path to the training pickle file')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save/load checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default='h0.pt', help='Checkpoint file name')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--shuffle', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False, help='Shuffle the dataset (True/False)')
    parser.add_argument('--input_dim', type=int, default=80, help='Input feature dimension')
    parser.add_argument('--encoder_dim', type=int, default=128, help='Encoder dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--num_attention_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')

    args = parser.parse_args()
    main(args)
