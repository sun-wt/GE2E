import os
import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

class BatchSampler:
    """
    BatchSampler 實現隨機選取 batch，其中數據可分為註冊與驗證部分。
    註冊部分用於中心點計算，驗證部分用於測試。
    """
    def __init__(self, data, batch_size=8, utterances_per_keyword=10):
        self.data = data
        self.batch_size = batch_size  # 每個 batch 包含的關鍵詞數量
        self.utterances_per_keyword = utterances_per_keyword  # 每個關鍵詞包含的樣本數量
        self.keywords = data['WORD'].unique()  # 已經過濾的關鍵詞

        print(f"Total valid keywords: {len(self.keywords)}")

    def __len__(self):
        """返回批次數量。"""
        return len(self.keywords) // self.batch_size

    def __iter__(self):
        while True:
            if len(self.keywords) < self.batch_size:
                raise ValueError(f"關鍵詞數量不足，僅有 {len(self.keywords)} 個關鍵詞。")

            # 隨機選擇 batch_size 個關鍵詞
            selected_keywords = list(np.random.choice(self.keywords, self.batch_size, replace=False))
            print(f"Selected keywords for batch: {selected_keywords}")

            # 構建批次數據
            batch_data = []
            for keyword in selected_keywords:
                keyword_data = self.data[self.data['WORD'] == keyword]
                reg_samples = keyword_data.sample(n=self.utterances_per_keyword, replace=False)

                # 分割為 5 個 enroll 和 5 個 valid
                enroll_samples = reg_samples.iloc[:self.utterances_per_keyword // 2]
                valid_samples = reg_samples.iloc[self.utterances_per_keyword // 2:]

                # 合併 enroll 和 valid
                batch_data.append(pd.concat([enroll_samples, valid_samples]))

            # 返回完整批次數據
            yield pd.concat(batch_data).reset_index(drop=True)


class MSWCDataloader(Sequence):
    """
    MSWC 的數據加載器，支持批次構造與特徵提取。
    """
    def __init__(self, 
                 batch_size=8,
                 fs=16000,
                 input_dim=80,
                 wav_dir='/datas/store162/syt/MST/MSWC/en/clips_wav',
                 csv_dir='/datas/store162/syt/MST/MSWC/en',
                 train_csv='en_train.csv',
                 test_csv='en_test.csv',
                 train=True,
                 shuffle=True,
                 pkl='/datas/store162/syt/GE2E/MSWC_MIN_10.pkl',
                 utterances_per_keyword=10):
        self.batch_size = batch_size
        self.fs = fs
        self.input_dim = input_dim
        self.wav_dir = wav_dir
        self.csv_dir = csv_dir
        self.csv_file = train_csv if train else test_csv
        self.shuffle = shuffle
        self.utterances_per_keyword = utterances_per_keyword
        self.pkl = pkl

        # 加載數據
        self.data = self._load_data()  # 確保 self.data 被加載後才初始化 BatchSampler

        # 初始化 BatchSampler
        self.batch_sampler = BatchSampler(
            self.data,
            batch_size=batch_size,
            utterances_per_keyword=utterances_per_keyword
        )

    def _load_data(self):
        """Load and preprocess data，並過濾掉樣本數不足的關鍵詞。"""
        if self.pkl and os.path.exists(self.pkl):
            print(f"Loading data from {self.pkl}")
            data = pd.read_pickle(self.pkl)
            print(f"Data loaded: {data.shape[0]} samples")
            return data

        # 原始加載流程
        print('Data loading...')
        data = []
        csv_file_path = os.path.join(self.csv_dir, self.csv_file)
        df = pd.read_csv(csv_file_path)

        for _, row in df.iterrows():
            wav_path = os.path.join(self.wav_dir, row['LINK'].replace(".opus", ".wav"))  # 轉換 .opus 為 .wav
            print(wav_path)
            anchor_text = row['WORD']
            if os.path.exists(wav_path):
                data.append({'wav': wav_path, 'WORD': anchor_text})

        data_df = pd.DataFrame(data)

        # 過濾掉樣本數不足的關鍵詞
        valid_keywords = data_df['WORD'].value_counts()
        valid_keywords = valid_keywords[valid_keywords >= self.utterances_per_keyword].index
        filtered_data = data_df[data_df['WORD'].isin(valid_keywords)]

        print(f"Filtered data: {len(filtered_data)} samples with {len(valid_keywords)} valid keywords.")

        # 保存為 .pkl 文件
        if self.pkl:
            print(f"Saving filtered data to {self.pkl}")
            filtered_data.to_pickle(self.pkl)

        return filtered_data

    def __len__(self):
        """返回批次數量。"""
        return len(self.batch_sampler)

    def __getitem__(self, idx):
        """生成批次數據。"""
        batch_data = next(iter(self.batch_sampler))
        batch_features = [self._extract_features(row['wav']) for _, row in batch_data.iterrows()]

        # 將關鍵詞映射為數字標籤
        label_to_index = {label: idx for idx, label in enumerate(sorted(batch_data['WORD'].unique()))}
        batch_labels = batch_data['WORD'].map(lambda x: label_to_index[x]).values

        pad_features = pad_sequences(batch_features, dtype='float32', padding='post')

        return np.array(pad_features), np.array(batch_labels)

    def _extract_features(self, wav_path):
        """提取 Mel 頻譜特徵。"""
        data, _ = librosa.load(wav_path, sr=self.fs)
        if len(data) < self.fs:
            data = np.pad(data, (0, self.fs - len(data)), mode='constant')
        else:
            data = data[:self.fs]
        mel_spec = librosa.feature.melspectrogram(y=data, sr=self.fs, n_mels=self.input_dim, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db.T  # 轉置為 (time_steps, input_dim)

    def on_epoch_end(self):
        """在每個 epoch 結束時打亂數據。"""
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)


if __name__ == "__main__":
    dataloader = MSWCDataloader(
        batch_size=8,
        input_dim=80,
        train=True,
        shuffle=True
    )
    for i in range(2):  # 打印前兩個批次
        print(f"\nBatch {i + 1}:")
        batch_features, batch_labels = dataloader[i]
        unique_labels, counts = np.unique(batch_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  Keyword {label}: {count} samples")
