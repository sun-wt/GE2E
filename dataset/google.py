import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import librosa

class GoogleCommandsDataloader(Sequence):
    def __init__(self, 
                 batch_size=8,
                 fs=16000,
                 input_dim=80,
                 wav_dir='/datas/store162/syt/GE2E/DB/speech_commands',
                 target_list=None,
                 shuffle=True,
                 utterances_per_keyword=10):
        self.batch_size = batch_size
        self.fs = fs
        self.input_dim = input_dim
        self.wav_dir = wav_dir
        self.target_list = target_list if target_list else []
        self.shuffle = shuffle
        self.utterances_per_keyword = utterances_per_keyword
        self.data = self._load_data()

        self.enrollment_data = {}
        self.test_data = {}
        self._split_data()

    def _load_data(self):
        """從目標列表加載數據，生成 DataFrame。"""
        data = []
        for target in self.target_list:
            target_dir = os.path.join(self.wav_dir, target)
            if os.path.exists(target_dir):  # 確保目錄存在
                wav_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.wav')]
                for wav in wav_files:
                    data.append({'wav': wav, 'keyword': target})
        return pd.DataFrame(data)

    def _split_data(self):
        """將數據分為 enrollment 和 test。"""
        for keyword in self.target_list:
            keyword_data = self.data[self.data['keyword'] == keyword]
            self.enrollment_data[keyword] = keyword_data.iloc[:self.utterances_per_keyword]
            self.test_data[keyword] = keyword_data.iloc[self.utterances_per_keyword:]

    def __len__(self):
        """返回總 batch 數。"""
        return len(self.target_list) // self.batch_size

    def __getitem__(self, idx):
        """生成批次數據。"""
        batch_keywords = self.target_list[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_features = []
        batch_labels = []
        for keyword in batch_keywords:
            # Enrollment 特徵
            enroll_features = [self._extract_features(row['wav']) for _, row in self.enrollment_data[keyword].iterrows()]
            # Test 特徵
            test_features = [self._extract_features(row['wav']) for _, row in self.test_data[keyword].iterrows()]

            # 添加到批次數據
            batch_features.extend(enroll_features + test_features)
            batch_labels.extend([keyword] * len(enroll_features) + [keyword] * len(test_features))

        # 對特徵進行補齊
        padded_features = pad_sequences(batch_features, dtype='float32', padding='post')
        label_to_index = {label: idx for idx, label in enumerate(sorted(self.target_list))}
        numeric_labels = [label_to_index[label] for label in batch_labels]

        return np.array(padded_features), np.array(numeric_labels)

    def _extract_features(self, wav_path):
        """提取 Mel spectrogram 特徵。"""
        data, _ = librosa.load(wav_path, sr=self.fs)
        if len(data) < self.fs:
            data = np.pad(data, (0, self.fs - len(data)), mode='constant')
        else:
            data = data[:self.fs]
        mel_spec = librosa.feature.melspectrogram(y=data, sr=self.fs, n_mels=self.input_dim, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db.T  # Transpose to (time_steps, input_dim)
    
    def _load_wav(self, wav_path):
        """載入音頻並提取 Mel spectrogram 特徵。"""
        data, _ = librosa.load(wav_path, sr=self.fs)
        
        # 補齊或截取音頻
        if len(data) < self.fs:
            data = np.pad(data, (0, self.fs - len(data)), mode='constant')
        else:
            data = data[:self.fs]

        # 計算 Mel spectrogram，形狀為 [n_mels, time_steps]
        mel_spec = librosa.feature.melspectrogram(y=data, sr=self.fs, n_mels=self.input_dim, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db.T  # 返回轉置後的 Mel spectrogram，形狀為 [time_steps, n_mels]


    def on_epoch_end(self):
        """在每個 epoch 結束時打亂數據。"""
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)


if __name__ == "__main__":
    target_list = [
        "up", "down", "left", "right", "yes", "no", "go", "stop", "on", "off"
    ]
    wav_dir = '/datas/store162/syt/GE2E/DB/speech_commands'

    loader = GoogleCommandsDataloader(
        batch_size=2,
        input_dim=80,
        wav_dir=wav_dir,
        target_list=target_list,
        shuffle=True,
        utterances_per_keyword=10
    )

    print("\nComplete Dataset Structure:")
    print(loader.data.head())
    print(loader.data.columns)

    for i in range(len(loader)):
        batch_features, batch_labels = loader[i]
        print(f"Batch {i + 1}:")
        print(f"Features shape: {batch_features.shape}")
        print(f"Labels: {batch_labels}")
