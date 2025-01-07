#!/usr/bin/env python3
import os
import pandas as pd
import pickle
import argparse
from tqdm import tqdm

def generate_mswc_pkl(csv_path, wav_dir, output_pkl, min_samples=10):
    """
    生成包含至少 min_samples 個樣本的單詞的 pickle 文件。

    Args:
        csv_path (str): CSV 文件的路徑。
        wav_dir (str): 存放 WAV 文件的目錄。
        output_pkl (str): 輸出 pickle 文件的路徑。
        min_samples (int): 單詞所需的最小樣本數量（默認為 10）。
    """
    print(f"載入 CSV 文件: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"無法讀取 CSV 文件: {e}")
        return

    print(f"CSV 加載完成，總行數: {df.shape[0]}")

    data_list = []
    missing_wavs = 0

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="處理數據"):
        if 'LINK' not in row or 'WORD' not in row:
            continue

        wav_relative_path = row['LINK'].replace(".opus", ".wav")
        wav_path = os.path.join(wav_dir, wav_relative_path)
        word = row['WORD']

        if not isinstance(word, str):
            continue

        if not os.path.exists(wav_path):
            missing_wavs += 1
            continue

        data_list.append({'wav': wav_path, 'WORD': word})

    print(f"收集到的樣本數量: {len(data_list)}")
    if missing_wavs > 0:
        print(f"缺少的 WAV 文件數量: {missing_wavs}")

    if not data_list:
        print("沒有有效的數據可保存，終止操作。")
        return

    # 轉換為 DataFrame 並篩選出至少 min_samples 個樣本的單詞
    df_data = pd.DataFrame(data_list)
    label_counts = df_data['WORD'].value_counts()
    valid_keywords = label_counts[label_counts >= min_samples].index
    filtered_df = df_data[df_data['WORD'].isin(valid_keywords)]

    print(f"篩選出具有至少 {min_samples} 個樣本的單詞後，樣本數量: {filtered_df.shape[0]}，唯一單詞數量: {len(valid_keywords)}")

    try:
        print(f"保存 pickle 文件到: {output_pkl}")
        filtered_df[['wav', 'WORD']].to_pickle(output_pkl)
        print("已成功保存 pickle 文件。")
    except Exception as e:
        print(f"保存 pickle 文件時出錯: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="從 CSV 生成包含至少指定數量樣本的 MSWC pickle 文件。")
    parser.add_argument('--csv_path', type=str, default='/datas/store162/syt/MST/MSWC/en/en_train.csv', help='CSV 文件的路徑')
    parser.add_argument('--wav_dir', type=str, default='/datas/store162/syt/MST/MSWC/en/clips_wav', help='存放 WAV 文件的目錄路徑')
    parser.add_argument('--output_pkl', type=str, default='MSWC_MIN_10_above10.pkl', help='輸出的 pickle 文件路徑')
    parser.add_argument('--min_samples', type=int, default=10, help='單詞的最小樣本數量（默認為 10）')

    args = parser.parse_args()

    generate_mswc_pkl(
        csv_path=args.csv_path,
        wav_dir=args.wav_dir,
        output_pkl=args.output_pkl,
        min_samples=args.min_samples
    )
