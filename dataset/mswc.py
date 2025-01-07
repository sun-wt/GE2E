import os
import numpy as np
import pandas as pd
import librosa
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

class BatchSampler:
    """
    與你原本的 BatchSampler 大致相同，
    只要能在 __iter__ 回傳一個批次的 DataFrame 即可。
    """
    def __init__(self, data, batch_size=8, utterances_per_keyword=10):
        self.data = data
        self.batch_size = batch_size
        self.utterances_per_keyword = utterances_per_keyword
        self.keywords = data['WORD'].unique()
        print(f"Total valid keywords: {len(self.keywords)}")

    def __len__(self):
        return len(self.keywords) // self.batch_size

    def __iter__(self):
        while True:
            if len(self.keywords) < self.batch_size:
                raise ValueError(f"關鍵詞數量不足，僅有 {len(self.keywords)} 個關鍵詞。")

            shuffled_keywords = random.sample(list(self.keywords), len(self.keywords))
            selected_keywords = shuffled_keywords[:self.batch_size]
            print(f"Selected keywords for batch: {selected_keywords}")

            batch_data = []
            for keyword in selected_keywords:
                keyword_data = self.data[self.data['WORD'] == keyword]
                if len(keyword_data) < self.utterances_per_keyword:
                    print(f"Skipping keyword '{keyword}' due to insufficient data.")
                    continue

                reg_samples = keyword_data.sample(n=self.utterances_per_keyword, replace=False)
                enroll_samples = reg_samples.iloc[:self.utterances_per_keyword // 2]
                valid_samples = reg_samples.iloc[self.utterances_per_keyword // 2:]
                batch_data.append(pd.concat([enroll_samples, valid_samples]))

            if len(batch_data) < self.batch_size:
                print("Insufficient data for batch, reselecting keywords...")
                continue

            yield pd.concat(batch_data).reset_index(drop=True)

class MSWCDataset(Dataset):
    """
    改用 PyTorch 自帶的 Dataset，而非 Keras 的 Sequence。
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
                 pkl_path='/datas/store162/syt/GE2E/MSWC_MIN_10.pkl',
                 utterances_per_keyword=10,
                 max_batches_per_epoch=100):
        self.batch_size = batch_size
        self.fs = fs
        self.input_dim = input_dim
        self.wav_dir = wav_dir
        self.csv_dir = csv_dir
        self.csv_file = train_csv if train else test_csv
        self.shuffle = shuffle
        self.utterances_per_keyword = utterances_per_keyword
        self.pkl = pkl_path
        self.max_batches_per_epoch = max_batches_per_epoch

        # 載入資料
        self.data = self._load_data()

        # 產生 batch sampler
        self.batch_sampler = iter(BatchSampler(
            self.data,
            batch_size=batch_size,
            utterances_per_keyword=utterances_per_keyword
        ))

    def _load_data(self):
        if self.pkl and os.path.exists(self.pkl):
            print(f"Loading data from {self.pkl}")
            try:
                data = pd.read_pickle(self.pkl)
                print(f"Data loaded: {data.shape[0]} samples")
                return data
            except Exception as e:
                print(f"Error loading {self.pkl}: {e}. Rebuilding data...")

        print('Loading data from CSV and extracting features...')
        data_list = []
        csv_file_path = os.path.join(self.csv_dir, self.csv_file)
        df = pd.read_csv(csv_file_path)

        temp_pkl = self.pkl + '.tmp'  # 臨時文件
        if os.path.exists(temp_pkl):
            print(f"Resuming from temporary file {temp_pkl}")
            try:
                data_list = pd.read_pickle(temp_pkl).to_dict('records')
            except Exception as e:
                print(f"Error loading temporary file {temp_pkl}: {e}. Starting fresh.")
                data_list = []

        for idx, row in df.iterrows():
            wav_path = os.path.join(self.wav_dir, row['LINK'].replace(".opus", ".wav"))
            if os.path.exists(wav_path):
                try:
                    # 確保 'WORD' 僅有一個單字
                    if isinstance(row['WORD'], str):
                        single_word = row['WORD'].split(',')[0].strip()  # 選擇第一個單字
                    else:
                        single_word = row['WORD']  # 假設是單一數值

                    # 提取特徵
                    features = self._extract_features(wav_path)
                    data_list.append({'features': features, 'WORD': single_word})
                    print(f"Processed {wav_path} with keyword '{single_word}'")

                    # 每處理1000個樣本，保存一次臨時文件
                    if len(data_list) % 1000 == 0:
                        print(f"Saving progress to temporary file {temp_pkl}")
                        pd.DataFrame(data_list).to_pickle(temp_pkl)
                except Exception as e:
                    print(f"Error processing {wav_path}: {e}")

        data_df = pd.DataFrame(data_list)
        valid_keywords = data_df['WORD'].value_counts()
        valid_keywords = valid_keywords[valid_keywords >= self.utterances_per_keyword].index
        filtered_data = data_df[data_df['WORD'].isin(valid_keywords)]

        print(f"Filtered data: {len(filtered_data)} samples with {len(valid_keywords)} valid keywords.")
        if self.pkl:
            print(f"Saving filtered data to {self.pkl}")
            filtered_data.to_pickle(self.pkl)
            os.remove(temp_pkl)  # 處理完成後刪除臨時文件

        return filtered_data

    def __len__(self):
        # 限定一個 epoch 最多做多少 batch
        return self.max_batches_per_epoch

    def __getitem__(self, idx):
        """
        每次從 batch_sampler 拿一個批次資料並組裝特徵和標籤。
        回傳結果要符合 (features, labels) 的格式給 DataLoader。
        """
        try:
            batch_data = next(self.batch_sampler)
        except StopIteration:
            self.batch_sampler = iter(BatchSampler(
                self.data,
                batch_size=self.batch_size,
                utterances_per_keyword=self.utterances_per_keyword
            ))
            batch_data = next(self.batch_sampler)

        batch_features = []
        batch_labels = []

        # 逐筆資料組裝特徵和標籤
        for _, row in batch_data.iterrows():
            feat = row['features']  # 直接從預先提取的特徵中獲取
            batch_features.append(torch.tensor(feat, dtype=torch.float32))
            batch_labels.append(row['WORD'])

        # pad_features: list of [time, input_dim]
        pad_features = pad_sequence(batch_features, batch_first=True)  # [batch_size*utterances_per_keyword, max_time, input_dim]

        # 建立 label mapping，確保相同單字有相同的label index
        unique_labels = sorted(list(set(batch_labels)))
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
        batch_labels_idx = torch.tensor([label_to_index[word] for word in batch_labels], dtype=torch.long)  # [batch_size * utterances_per_keyword]

        # 在這裡加上通道維度 [batch_size * utterances_per_keyword, 1, max_time, input_dim]
        pad_features = pad_features.unsqueeze(1)

        # print(f"MSWCDataset __getitem__ [{idx}]: inputs.shape = {pad_features.shape}, labels.shape = {batch_labels_idx.shape}")
        # print(f"  inputs.min()={pad_features.min().item()}, inputs.max()={pad_features.max().item()}, inputs.mean()={pad_features.mean().item()}, inputs.std()={pad_features.std().item()}")

        return pad_features, batch_labels_idx

    def _extract_features(self, wav_path, fs=16000, input_dim=80):
        if not isinstance(wav_path, str):
            raise ValueError(f"Invalid file: {wav_path}")

        data, _ = librosa.load(wav_path, sr=fs)
        if len(data) < fs:
            data = np.pad(data, (0, fs - len(data)), mode='constant')
        else:
            data = data[:fs]

        # 計算音頻能量
        energy = np.sum(data ** 2) / len(data)
        # print(f"_extract_features: wav_path={wav_path}, Energy={energy}")

        if energy < 1e-6:
            # print(f"  Warning: Low energy in {wav_path}, adjusting...")
            # 調整音頻信號（例如增加音量）
            data = data * (1e-6 / energy) ** 0.5

        mel_spec = librosa.feature.melspectrogram(y=data, sr=fs, n_mels=input_dim, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=100)  # 增加 top_db

        # 打印統計信息（標準化前）
        # print(f"  Mel-Spectrogram shape: {mel_spec.shape}")
        # print(f"  Mel-Spectrogram dB min={mel_spec_db.min()}, max={mel_spec_db.max()}")

        # 標準化
        mel_spec_db_normalized = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-9)

        # 打印統計信息（標準化後）
        # print(f"  After normalization: min={mel_spec_db_normalized.min()}, max={mel_spec_db_normalized.max()}, mean={mel_spec_db_normalized.mean()}, std={mel_spec_db_normalized.std()}")

        # 可視化標準化前後的梅爾頻譜圖
        # self.visualize_features(mel_spec_db, mel_spec_db_normalized, wav_path)

        return mel_spec_db_normalized.T

    def visualize_features(self, mel_spec_db, mel_spec_db_normalized, wav_path):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].imshow(mel_spec_db, aspect='auto', origin='lower', cmap='viridis')
        axs[0].set_title('Mel-Spectrogram (dB)')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Mel Bands')

        axs[1].imshow(mel_spec_db_normalized, aspect='auto', origin='lower', cmap='viridis')
        axs[1].set_title('Mel-Spectrogram (Normalized)')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Mel Bands')

        plt.suptitle(f"Visualization of Mel-Spectrogram for {os.path.basename(wav_path)}")
        plt.tight_layout()
        plt.show()

    def on_epoch_end(self):
        # 若有需要洗牌
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
