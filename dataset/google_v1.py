import os
import numpy as np
import pandas as pd
import librosa
import pickle
from pathlib import Path

class GoogleCommandsDataloader:
    def __init__(self, 
                 fs=16000,
                 input_dim=80,
                 wav_dir='/datas/store162/syt/GE2E/DB/google_speech_commands',
                 target_list=None,
                 utterances_per_keyword=10):
        self.fs = fs
        self.input_dim = input_dim
        self.wav_dir = wav_dir
        self.target_list = target_list if target_list else []
        self.utterances_per_keyword = utterances_per_keyword
        self.data = self._load_data()

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

    def split_data_and_save(self, enroll_file='enroll.pkl', test_file='test.pkl'):
        """將數據分為 enrollment 和 test 並保存為 .pkl 檔案。"""
        enrollment_data = {}
        test_data = {}

        for keyword in self.target_list:
            keyword_data = self.data[self.data['keyword'] == keyword]
            enroll_wavs = keyword_data.iloc[:self.utterances_per_keyword]['wav'].tolist()
            test_wavs = keyword_data.iloc[self.utterances_per_keyword:]['wav'].tolist()

            # 提取特徵
            enrollment_data[keyword] = [self._extract_features(wav) for wav in enroll_wavs]
            test_data[keyword] = [self._extract_features(wav) for wav in test_wavs]

        # 保存到 .pkl 文件
        with open(enroll_file, 'wb') as f:
            pickle.dump(enrollment_data, f)
            print(f"Enrollment data saved to {enroll_file}")

        with open(test_file, 'wb') as f:
            pickle.dump(test_data, f)
            print(f"Test data saved to {test_file}")


if __name__ == "__main__":
    wav_dir = "/datas/store162/syt/GE2E/DB/google_speech_commands"
    all_targets = [
        d.name for d in Path(wav_dir).iterdir()
        if d.is_dir() and d.name not in ["_background_noise_"]
    ]

    print(f"Target List: {all_targets}")

    loader = GoogleCommandsDataloader(
        input_dim=80,
        wav_dir=wav_dir,
        target_list=all_targets,
        utterances_per_keyword=10
    )

    # 分割數據並保存
    loader.split_data_and_save(enroll_file='enroll.pkl', test_file='test.pkl')
