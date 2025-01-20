import os
import torch
import torch.quantization
from conformer.conformer.encoder import ConformerEncoder
from tiny_conformer.conformer.model_def import ConformerEncoder as TinyConformerEncoder

def get_model_size(file_path):
    """計算模型文件的大小（以 MB 為單位）"""
    file_size_mb = os.path.getsize(file_path) / (1024 ** 2)
    return file_size_mb

# 加載模型
# model = ConformerEncoder(input_dim=40, encoder_dim=256, num_layers=2, num_attention_heads=2)
checkpoint_path = "checkpoints/tiny_40_512_12_8.pt"
model = TinyConformerEncoder(input_dim=40,num_layers=12)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

# 打印原始模型大小
original_model_path = "model_original.pt"
torch.save(model, original_model_path)
original_model_size = get_model_size(original_model_path)
print(f"原始模型大小: {original_model_size:.2f} MB")

# # 動態量化
# quantized_model = torch.quantization.quantize_dynamic(
#     model, {torch.nn.Linear}, dtype=torch.qint8
# )

# # 保存整個量化模型
# quantized_model_path = "model_quantized.pt"
# torch.save(quantized_model, quantized_model_path)

# # 打印量化後的模型大小
# quantized_model_size = get_model_size(quantized_model_path)
# print(f"量化後模型大小: {quantized_model_size:.2f} MB")

# # 嘗試加載量化後的模型
# try:
#     loaded_model = torch.load(quantized_model_path, map_location="cpu")
#     loaded_model.eval()
#     print("量化後的模型加載成功！")
# except Exception as e:
#     print(f"加載量化模型時發生錯誤: {e}")
