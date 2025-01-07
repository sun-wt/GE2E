import os
import torch
from torch.utils.data import DataLoader
import pickle
from conformer.conformer.model import Conformer
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

# Dataset 定義
class FixedTrainDataset(torch.utils.data.Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.all_data = pickle.load(f)
        self.features = [torch.tensor(item[0], dtype=torch.float32) for item in self.all_data]
        self.labels = [item[1] for item in self.all_data]

        # 將數據按單字分組
        self.word_to_samples = defaultdict(list)
        for feature, label in zip(self.features, self.labels):
            self.word_to_samples[label].append(feature)

        # 確保每個單字的樣本數量滿足條件
        for word, samples in self.word_to_samples.items():
            if len(samples) != 10:
                raise ValueError(f"單字 '{word}' 的樣本數量不是 10，而是 {len(samples)}！")

        # 按照 8 個單字為一組構建批次
        self.batches = []
        words = list(self.word_to_samples.keys())
        for i in range(0, len(words), 8):  # 每次選擇 8 個單字
            batch_words = words[i:i + 8]
            batch = []
            for word in batch_words:
                batch.extend(self.word_to_samples[word])  # 每個單字 10 個樣本
            self.batches.append((batch, batch_words))  # 保存特徵和對應單字

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch, words = self.batches[idx]
        return batch, words


# 批次處理函數
def collate_fn(batch):
    features, words = batch[0]
    features = [torch.tensor(f, dtype=torch.float32) for f in features]
    padded_features = pad_sequence(features, batch_first=True)  # 將特徵補齊到相同長度
    return padded_features, words


def check_batches_with_words(pkl_path, model, batch_size, device):
    """
    檢查每個批次中的數據質量，按單字分組並輸出統計信息。
    """
    dataset = FixedTrainDataset(pkl_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()

    print("\n[Batch-wise Word-level Inspection]")
    for batch_idx, (inputs, words) in enumerate(dataloader):
        # 移動數據到 GPU / CPU
        inputs = inputs.to(device)

        # Forward through Conformer
        input_lengths = torch.tensor([inputs.shape[1]] * inputs.shape[0], dtype=torch.long, device=device)
        embeddings, _ = model(inputs, input_lengths)

        # 收集 embeddings 和 labels
        embeddings = embeddings.cpu().detach().numpy()  # [batch_size, time, embedding_dim]

        print(f"\nBatch {batch_idx + 1}:")
        for i, word in enumerate(words):
            samples = embeddings[i * 10:(i + 1) * 10]  # 每個單字的 10 個樣本
            samples = torch.tensor(samples)  # 轉換為 Tensor
            print(f"  Word '{word}':")
            print(f"    Samples shape: {samples.shape}")
            print(f"    Min: {samples.min().item():.4f}, Max: {samples.max().item():.4f}")
            print(f"    Mean: {samples.mean().item():.4f}, Std: {samples.std().item():.4f}")


if __name__ == "__main__":
    # 加載模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conformer(
        num_classes=35,  # 請根據實際設置
        input_dim=80,
        encoder_dim=128,
        num_encoder_layers=2,
        num_attention_heads=4,
    ).to(device)

    checkpoint_path = '/datas/store162/syt/GE2E/checkpoints/h0.pt'
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"[Info] 初始模型權重 (h0) 已從 {checkpoint_path} 加載。")
    else:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[Info] 初始模型權重 (h0) 已保存至 {checkpoint_path}")

    # 檢查批次數據
    check_batches_with_words(
        pkl_path="generate/train_fixed.pkl",
        model=model,
        batch_size=1,  # 每次加載一個批次 (8 個單字，每個單字 10 個樣本)
        device=device,
    )
