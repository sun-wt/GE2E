import torch.nn as nn
import os
import torch

class ThreeLayerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        """
        初始化三層 LSTM 架構
        Args:
            input_dim (int): 輸入特徵維度
            hidden_dim (int): 隱藏層維度
            output_dim (int): 最終輸出特徵維度
            dropout (float): dropout 機率
        """
        super(ThreeLayerLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=3,  # 固定為三層
            batch_first=True,
            bidirectional=False,  # 單向 LSTM
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)  # 最終全連接層，用於調整輸出維度

    def forward(self, inputs, input_lengths):
        """
        前向傳播
        Args:
            inputs (torch.Tensor): 輸入序列，形狀為 [batch_size, time_steps, input_dim]
            input_lengths (torch.Tensor): 每個序列的有效長度
        Returns:
            torch.Tensor: LSTM 輸出的最終特徵 [batch_size, time_steps, output_dim]
        """
        # 必須將 input_lengths 轉為 CPU int64，這是 PyTorch 的限制
        input_lengths = input_lengths.cpu().to(torch.int64)

        # 使用 pack_padded_sequence 處理填充序列
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            inputs, input_lengths, batch_first=True, enforce_sorted=False
        )
        packed_outputs, _ = self.lstm(packed_inputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        outputs = self.fc(outputs)  # 通過全連接層調整輸出維度
        return outputs, input_lengths


if __name__ == "__main__":
    # 初始化模型
    input_dim = 80  # 輸入特徵維度
    hidden_dim = 256  # 隱藏層維度
    output_dim = 128  # 最終輸出特徵維度
    dropout = 0.1  # dropout 機率
    model = ThreeLayerLSTM(input_dim, hidden_dim, output_dim, dropout)

    # 確保目標儲存資料夾存在
    save_dir = '/datas/store162/syt/GE2E/checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # 儲存模型
    save_path = os.path.join(save_dir, 'model_lstm.pt')
    torch.save(model.state_dict(), save_path)

    print(f"模型已成功儲存至 {save_path}")
