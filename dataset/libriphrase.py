import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import librosa


class BatchSampler:
    """
    BatchSampler 實現隨機選取 batch，其中數據可分為註冊與驗證部分。
    註冊部分用於中心點計算，驗證部分用於測試。
    """
    def __init__(self, data, batch_size=8, utterances_per_keyword=10, split_ratio=0.5):
        self.data = data
        self.batch_size = batch_size  # 每個 batch 包含的 keyword 數量
        self.utterances_per_keyword = utterances_per_keyword  # 每個 keyword 包含的 utterance 數量
        self.keywords = data['anchor_text'].unique()  # 獲取唯一的 anchor_text 作為關鍵字

        # 按照 split_ratio 分割數據為註冊和驗證部分
        self.registration_data = {}
        self.validation_data = {}
        for keyword in self.keywords:
            keyword_data = data[data['anchor_text'] == keyword]
            split_point = int(len(keyword_data) * split_ratio)
            self.registration_data[keyword] = keyword_data.iloc[:split_point]
            self.validation_data[keyword] = keyword_data.iloc[split_point:]

    def __len__(self):
        """
        返回可生成的 batch 數。
        """
        return len(self.keywords) // self.batch_size

    def __iter__(self):
        while True:
            # 確保選取的 keyword 滿足 batch_size 的需求
            if len(self.keywords) < self.batch_size:
                raise ValueError(f"Number of keywords ({len(self.keywords)}) is less than batch size ({self.batch_size}).")

            selected_keywords = list(np.random.choice(self.keywords, self.batch_size, replace=False))

            # 打印選取的關鍵字
            print(f"Selected keywords: {selected_keywords}")

            # 構建批次數據
            batch_data = []
            for keyword in selected_keywords:
                reg_samples = self.registration_data[keyword].sample(
                    n=self.utterances_per_keyword // 2,
                    replace=(len(self.registration_data[keyword]) < self.utterances_per_keyword // 2)
                )
                val_samples = self.validation_data[keyword].sample(
                    n=self.utterances_per_keyword // 2,
                    replace=(len(self.validation_data[keyword]) < self.utterances_per_keyword // 2)
                )

                # 合併註冊與驗證樣本
                batch_data.append(pd.concat([reg_samples, val_samples]))

            # 返回完整批次數據
            yield pd.concat(batch_data).reset_index(drop=True)


class LibriPhraseLoader(Sequence):
    def __init__(self, 
                 batch_size,
                 fs=16000,
                 input_dim=80,
                 wav_dir='/datas/store162/syt/PhonMatchNet/DB/LibriPhrase/wav_dir',
                 csv_dir='/datas/store162/syt/PhonMatchNet/DB/LibriPhrase/csv_dir',
                 train_csv=['train_100h', 'train_360h'],
                 test_csv=['train_500h'],
                 pkl=None,
                 train=True,
                 shuffle=True,
                 utterances_per_keyword=10):
        self.batch_size = batch_size
        self.fs = fs
        self.input_dim = input_dim
        self.wav_dir = wav_dir
        self.csv_dir = csv_dir
        self.train_csv = train_csv if train else test_csv
        self.train = train
        self.shuffle = shuffle
        self.pkl = pkl
        self.utterances_per_keyword = utterances_per_keyword
        self.data = self._load_data()

        # 初始化 BatchSampler
        self.batch_sampler = BatchSampler(
            self.data,
            batch_size=batch_size,
            utterances_per_keyword=utterances_per_keyword
        )

    def _load_data(self):
        """Load and preprocess data."""
        if self.pkl and os.path.exists(self.pkl):
            print(f"Loading data from {self.pkl}")
            data = pd.read_pickle(self.pkl)
            print(f"Data loaded: {data.shape[0]} samples")
            print(data.head())
            return data

        # 原始加載流程
        data = []
        for db in self.train_csv:
            csv_files = [os.path.join(self.csv_dir, f) for f in os.listdir(self.csv_dir) if db in f]
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    wav_path = os.path.join(self.wav_dir, row['anchor'])
                    anchor_text = row['anchor_text']
                    if os.path.exists(wav_path):
                        data.append({'wav': wav_path, 'anchor_text': anchor_text})
        
        data_df = pd.DataFrame(data)

        # 分割數據集
        split_ratio = 0.5  # 設置分割比例
        grouped = data_df.groupby('anchor_text')
        train_data = grouped.apply(lambda x: x.sample(frac=split_ratio)).reset_index(drop=True)

        if self.pkl:
            print(f"Saving data to {self.pkl}")
            train_data.to_pickle(self.pkl)

        return train_data

    def __len__(self):
        """返回總 batch 數。"""
        return len(self.batch_sampler)

    # def __getitem__(self, idx):
    #     batch_data = next(iter(self.batch_sampler))
    #     batch_features = [self._extract_features(row['wav']) for _, row in batch_data.iterrows()]
    #     batch_labels = batch_data['anchor_text'].values
    #     pad_features = pad_sequences(batch_features, dtype='float32', padding='post')
        
    #     # 打印批次的特徵和標籤資訊
    #     print(f"Batch {idx}: Features shape {np.array(pad_features).shape}, Labels shape {np.array(batch_labels).shape}")
        
    #     # 去重後統計關鍵字數量
    #     unique_labels = np.unique(batch_labels)
    #     for label in unique_labels:
    #         count = (batch_labels == label).sum()
    #         # print(f"  Keyword {label}: {count} samples")

    #     return np.array(pad_features), np.array(batch_labels)
    
    def __getitem__(self, idx):
        batch_data = next(iter(self.batch_sampler))
        batch_features = [self._extract_features(row['wav']) for _, row in batch_data.iterrows()]
        
        # 使用數字標籤而非字串標籤
        label_to_index = {label: idx for idx, label in enumerate(sorted(set(batch_data['anchor_text'])))}
        batch_labels = batch_data['anchor_text'].map(lambda x: label_to_index[x]).values
        
        pad_features = pad_sequences(batch_features, dtype='float32', padding='post')
        
        # 打印批次的特徵和標籤資訊
        # print(f"Batch {idx}: Features shape {np.array(pad_features).shape}, Labels shape {np.array(batch_labels).shape}")
        
        return np.array(pad_features), np.array(batch_labels)


    def _extract_features(self, wav_path):
        """Extract Mel spectrogram features from audio."""
        data, _ = librosa.load(wav_path, sr=self.fs)
        if len(data) < self.fs:
            data = np.pad(data, (0, self.fs - len(data)), mode='constant')
        else:
            data = data[:self.fs]
        mel_spec = librosa.feature.melspectrogram(y=data, sr=self.fs, n_mels=self.input_dim, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db.T  # Transpose to (time_steps, input_dim)

    def on_epoch_end(self):
        """每個 epoch 結束時打亂數據。"""
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)


if __name__ == "__main__":
    train_dataloader = LibriPhraseLoader(
        batch_size=8,
        input_dim=80,
        train=True,
        shuffle=True
    )
    for i in range(2):  # 打印前兩個批次
        print(f"\nBatch {i + 1}:")
        batch_features, batch_labels = train_dataloader[i]
        unique_labels, counts = np.unique(batch_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  Keyword {label}: {count} samples")

    
    # from datasets import load_dataset

    # # 加載 MSWC 的英語部分
    # dataset = load_dataset("MLCommons/ml_spoken_words", "en_wav")

    # # 查看數據集的基本信息
    # print(dataset)



