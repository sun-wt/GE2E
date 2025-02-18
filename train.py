import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from conformer.conformer.encoder import ConformerEncoder
from tiny_conformer.conformer.encoder import ConformerEncoder as TinyConformerEncoder
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
import glob
from Levenshtein import distance as levenshtein_distance
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import torchaudio.transforms as T

# 你自己的噪音資料夾路徑 (sound-bible)
NOISE_DIR = "/datas/store162/syt/PhonMatchNet/DB/sound-bible"
SPEC_AUG_PROB = 0.3       # 是否對梅爾頻譜做 SpecAugment 的機率
TIME_MASK_PARAM = 80      # TimeMasking 遮罩參數，可依時間維度大小調整
FREQ_MASK_PARAM = 30      # FrequencyMasking 遮罩參數，可依頻率維度大小調整

# 設定設備
print(f"PyTorch 版本: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============= 新增：LSTM 模型類別 =============
class ThreeLayerLSTM(nn.Module):
    """
    以三層 LSTM 為骨幹，並在最後做 Global Average Pooling，
    回傳 shape = [batch, hidden_dim] 的向量，供 GE2E Loss 使用。
    """
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super(ThreeLayerLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        # 這裡可以再加上一個全連接層或 LayerNorm, 依照需求調整
        # e.g. self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inputs, input_lengths):
        """
        inputs: [batch, time, input_dim]
        input_lengths: [batch]，實際有效序列長度
        回傳:
            emb: [batch, hidden_dim]
            input_lengths: [batch]
        """
        # pack_padded_sequence 需要用 CPU 上的 long tensor
        cpu_lens = input_lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            inputs, cpu_lens, batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.lstm(packed)  # outputs shape: [batch, time, hidden_dim] (packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # 簡單做個 Global Average Pooling，把 time 維度平均
        emb = torch.mean(outputs, dim=1)  # [batch, hidden_dim]

        return emb, input_lengths
# ============= 新增：LSTM 模型類別 End =============

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
            
        # ========== 使用機率判斷是否加噪音 ==========
        if random.random() < NOISE_PROBABILITY:
            # 1) 計算語音訊號 RMS
            rms_signal = np.sqrt(np.mean(data**2)) + 1e-9

            # 2) 隨機產生一個 SNR 值 (3~15 dB)
            snr_dB = random.uniform(3, 15)

            # 3) 由 SNR 計算所需 noise 的 RMS
            #    snr_dB = 20 * log10(rms_signal / rms_noise)
            #    => rms_noise = rms_signal / 10^(snr_dB/20)
            rms_noise = rms_signal / (10 ** (snr_dB / 20))

            # 4) 從資料夾隨機挑選一個噪音檔
            noise_files = glob.glob(os.path.join(NOISE_DIR, "*.wav"))
            if noise_files:
                noise_file = random.choice(noise_files)
                noise_data, _ = librosa.load(noise_file, sr=fs)

                # 若噪音檔比 1 秒長，就隨機裁切一段 1 秒；否則 zero-padding
                if len(noise_data) < fs:
                    noise_data = np.pad(noise_data, (0, fs - len(noise_data)), mode='constant')
                else:
                    start_idx = random.randint(0, len(noise_data) - fs)
                    noise_data = noise_data[start_idx : start_idx + fs]

                # 計算 noise_data RMS，並調整到目標值
                current_noise_rms = np.sqrt(np.mean(noise_data**2)) + 1e-9
                noise_data = noise_data * (rms_noise / current_noise_rms)

                # 混合
                data = data + noise_data

        # ==================================

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

        # 執行 CMVN 標準化 (均值方差正規化)
        # mean = mel_spec_db.mean()
        # std = mel_spec_db.std()
        # mel_spec_db_normalized = (mel_spec_db - mean) / (std + 1e-9)

        return mel_spec_db_normalized.T  # [time, n_mels]
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

def get_features(audio_file, fs=16000, input_dim=40):
    """
    Extract MFCC features from an audio file and return numpy array with output shape=(TIME, MFCC).
    input - audio file path
    output - MFCC tensor
    """
    # Load audio using Librosa
    waveform, sample_rate = librosa.load(audio_file, sr=None)

    # Check whether audio is mono channel otherwise convert to mono
    if len(waveform.shape) == 2:
        waveform = librosa.to_mono(waveform.transpose())

    # Resample to 16kHz
    if sample_rate != 16000:
        waveform = librosa.resample(y=waveform, orig_sr=sample_rate, target_sr=fs)


    # Get Mel Frequency Cepstral Coefficients
    features = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=input_dim)

    return features.transpose()

# Dataset 定義
class DynamicTrainDataset(Dataset):
    """
    動態從 MSWC_MIN_10.pkl 提取 wav 文件並轉換為特徵。
    - `hard_samples` 模式下：
        1. 隨機選擇 **1 個主要關鍵詞**
        2. 找到 `batch_size - 1` 個發音相似的關鍵詞作為難負樣本
        3. 從每個關鍵詞中隨機選擇 `samples_per_label` 個音檔
    """

    def __init__(self, pkl_path, batch_size=8, samples_per_label=10, input_dim=80, fs=16000, virtual_length=10000, hard_samples=False, threshold=2):
        # 載入數據（DataFrame 格式）
        self.data = pd.read_pickle(pkl_path)
        if not {'wav', 'WORD'}.issubset(self.data.columns):
            raise ValueError("The DataFrame 必須包含 'wav' 和 'WORD' 欄位。")
        
        self.batch_size = batch_size
        self.samples_per_label = samples_per_label
        self.input_dim = input_dim
        self.fs = fs
        self.virtual_length = virtual_length
        self.hard_samples = hard_samples  # 是否選擇難樣本
        self.threshold = threshold  # Levenshtein 距離閾值
        
        # 使用 groupby 快速分組生成字典
        print("Grouping data by WORD...")
        self.word_to_samples = self.data.groupby('WORD')['wav'].apply(list).to_dict()
        self.words = list(self.word_to_samples.keys())  # 取得所有關鍵詞列表

    def _find_hard_negatives(self, target_word):
        """
        找到與目標關鍵詞發音相似的 `batch_size - 1` 個關鍵詞作為難負樣本。
        """
        similar_words = [w for w in self.words if w != target_word and levenshtein_distance(target_word, w) <= self.threshold]
        
        # 若發音相似的詞不夠 `batch_size - 1`，則補充隨機關鍵詞
        if len(similar_words) < self.batch_size - 1:
            extra_words = [w for w in self.words if w not in similar_words and w != target_word]
            similar_words.extend(random.sample(extra_words, min(len(extra_words), self.batch_size - 1 - len(similar_words))))

        return random.sample(similar_words, min(len(similar_words), self.batch_size - 1))

    def __len__(self):
        return self.virtual_length

    def __getitem__(self, idx):
        valid_words = self.words  # 直接使用關鍵詞列表

        if self.hard_samples:
            # **隨機選擇 1 個主要關鍵詞**
            main_word = random.choice(valid_words)

            # **找到 `batch_size - 1` 個發音相似的難負樣本**
            hard_neg_words = self._find_hard_negatives(main_word)
            selected_words = [main_word] + hard_neg_words
        else:
            # **標準模式：隨機選擇 `batch_size` 個關鍵詞**
            selected_words = random.sample(valid_words, self.batch_size)
        
        print(f"選擇關鍵詞: {selected_words}")

        batch_features = []
        batch_labels = []

        for label_idx, word in enumerate(selected_words):
            samples = self.word_to_samples[word]
            selected_samples = random.sample(samples, self.samples_per_label)

            # 加載音檔並提取特徵
            for wav_path in selected_samples:
                if not os.path.exists(wav_path):
                    print(f"[Warning] WAV 文件不存在: {wav_path}")
                    continue

                features = extract_features(wav_path, fs=self.fs, input_dim=self.input_dim)
                # features = get_features(wav_path, fs=self.fs, input_dim=self.input_dim)
                if features is not None:
                    batch_features.append(torch.tensor(features, dtype=torch.float32))
                    batch_labels.append(label_idx)

        pad_features = pad_sequence(batch_features, batch_first=True)  
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)  

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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0.0
        all_auc = []

        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            input_lengths = torch.tensor([inputs.shape[1]] * inputs.shape[0], dtype=torch.long, device=device)

            # 檢查 NaN 或 Inf
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print(f"[Error] Step {step}: inputs 出現 NaN 或 Inf")
                continue

            with autocast():
                emb, _ = model(inputs, input_lengths)
                
                # 檢查 embeddings 是否有 NaN 或 Inf
                if torch.isnan(emb).any() or torch.isinf(emb).any():
                    print(f"[Error] Step {step}: emb 出現 NaN 或 Inf")
                    continue
                
                loss, auc = loss_fn(emb, labels)  # 返回 Loss 和 AUC

                # 檢查 loss 是否 NaN 或 Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[Error] Step {step}: loss 出現 NaN 或 Inf")
                    continue
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 限制梯度大小
            
            # 檢查梯度是否 NaN 或 Inf
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"[Error] 梯度 NaN/Inf: {name}")
                        continue
            
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            all_auc.append(auc)

            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}, AUC: {auc:.4f}")
                print(f"  -> alpha={loss_fn.alpha.item():.4f}, beta={loss_fn.beta.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        avg_auc = sum(all_auc) / len(all_auc) if all_auc else 0.0  # 避免除零錯誤
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}, Average AUC: {avg_auc:.4f}")

        scheduler.step(avg_loss)  # 如果 loss 沒改善，調降學習率

        checkpoint_path = os.path.join(save_checkpoint_dir, f"epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[Info] 模型檢查點已保存至 {checkpoint_path}")

def main(args):
    # 加載數據集
    global NOISE_PROBABILITY  # 顯式聲明全域變數
    NOISE_PROBABILITY = args.noise_prob  # 設定 NOISE_PROBABILITY
    
    dataset = DynamicTrainDataset(
        pkl_path=args.pkl_path,
        batch_size=args.batch_size,
        samples_per_label=args.samples_per_label,
        input_dim=args.input_dim,
        virtual_length=args.virtual_length,
        hard_samples=args.hard_samples,  # 是否啟用難樣本
        threshold=args.threshold  # Levenshtein 距離閾值
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    # 初始化模型
    if args.model_type == "tiny":
        print("[Info] 使用 Tiny Conformer 模型")
        model = TinyConformerEncoder(
            input_dim=args.input_dim,
            encoder_dim=args.encoder_dim,
            num_layers=args.num_encoder_layers,
            num_attention_heads=args.num_attention_heads
        ).to(device)
    elif args.model_type == "normal":
        print("[Info] 使用普通 Conformer 模型")
        model = ConformerEncoder(
            input_dim=args.input_dim,
            encoder_dim=args.encoder_dim,
            num_layers=args.num_encoder_layers,
            num_attention_heads=args.num_attention_heads
        ).to(device)
    elif args.model_type == "lstm":
        print("[Info] 使用三層 LSTM 模型")
        model = ThreeLayerLSTM(
            input_dim=args.input_dim,
            hidden_dim=args.encoder_dim,  # 直接使用 encoder_dim 當作 LSTM hidden_dim
            dropout=0.1
        ).to(device)
    else:
        raise ValueError("不支援的 model_type，請使用 normal / tiny / lstm")

    # 打印模型結構
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

    alpha, beta = args.alpha_beta
    use_alpha_beta = not (alpha == 0.0 and beta == 0.0)
    print(f"[Info] use_alpha_beta: {use_alpha_beta}, alpha: {alpha}, beta: {beta}")
    
    # 若啟用 Fine-tuning，則載入預訓練模型
    if args.tune:
        rate = 0.7  # 解凍比例
        for param in list(model.parameters())[:int(len(list(model.parameters())) * rate)]:
            param.requires_grad = False
        print("[Info] 只微調 Conformer Encoder 的後", rate*100, "層")
        
        
    # 設置損失函數和優化器
    loss_fn = GE2ELoss(device=device, use_alpha_beta=use_alpha_beta, init_alpha=alpha, init_beta=beta)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.Adam([
    #     {'params': model.parameters(), 'lr': 1e-3},
    #     {'params': loss_fn.parameters(), 'lr': 1e-4}#對 GE2ELoss 的 alpha, beta 設置更大的 lr
    # ])


    # 創建檢查點保存目錄
    os.makedirs(args.save_dir, exist_ok=True)

    # 訓練模型
    train_model(model, dataloader, optimizer, loss_fn, device, args.epochs, args.save_dir)


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
    parser.add_argument('--model_type', type=str, default="normal", choices=["normal", "tiny"], help="Specify the model type (normal or tiny)")
    parser.add_argument('--alpha_beta', type=float, nargs=2, metavar=('ALPHA', 'BETA'),
                        default=[0.0, 0.0],
                        help='Set alpha and beta values for GE2E loss. Use "0 0" to disable alpha-beta scaling.')
    
    # Fine-tuning 開關
    parser.add_argument('--tune', action='store_true', help="啟用 Fine-tuning，載入預訓練模型並鎖定部分層")
    parser.add_argument('--noise_prob', type=float, default=0.2, help="噪音添加機率")
    parser.add_argument('--hard_samples', action='store_true', help="啟用難正負樣本選擇模式")
    parser.add_argument('--threshold', type=int, default=2, help="Levenshtein 距離閾值")
    
    args = parser.parse_args()
    
    main(args)