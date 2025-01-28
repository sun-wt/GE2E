import torch
from tiny_conformer.conformer.model_def import ConformerEncoder as TinyConformerEncoder
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 計算 PyTorch 模型參數大小
def get_pytorch_model_size(model):
    """計算 PyTorch 模型的實際大小（以 MB 為單位）"""
    total_params = sum(p.numel() for p in model.parameters())  # 總參數數量
    param_size_in_bytes = total_params * 4  # 假設每個參數佔用 4 字節（float32）
    return param_size_in_bytes / (1024 ** 2)  # 轉換為 MB

# 計算 TensorFlow 模型參數大小
def get_tensorflow_model_size(saved_model_dir):
    """計算 TensorFlow 模型的實際大小（以 MB 為單位）"""
    total_params = 0
    model = tf.saved_model.load(saved_model_dir)
    
    # 檢查是否有 'trainable_variables' 或 'variables'
    if hasattr(model, 'variables'):
        for variable in model.variables:
            total_params += tf.size(variable).numpy()
    elif hasattr(model, 'trainable_variables'):
        for variable in model.trainable_variables:
            total_params += tf.size(variable).numpy()

    # 假設每個參數佔用 4 字節（float32）
    param_size_in_bytes = total_params * 4
    return param_size_in_bytes / (1024 ** 2)  # 轉換為 MB

# Step 1: PyTorch 模型 -> ONNX
def convert_to_onnx():
    model = TinyConformerEncoder(input_dim=40, encoder_dim=256, num_layers=2, num_attention_heads=2)
    model.eval()

    dummy_input = torch.randn(1, 100, 40)  # 假設輸入形狀
    dummy_input_lengths = torch.tensor([100])  # 輸入長度

    # 打印 PyTorch 模型參數大小
    print(f"[Info] PyTorch 模型參數大小: {get_pytorch_model_size(model):.2f} MB")

    onnx_model_path = "conformer_model.onnx"
    torch.onnx.export(
        model,
        (dummy_input, dummy_input_lengths),
        onnx_model_path,
        export_params=True,
        opset_version=11,
        input_names=["input", "input_lengths"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "time"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time"}
        }
    )
    print(f"[Info] PyTorch 模型已轉換為 ONNX 格式並保存至: {onnx_model_path}")
    return onnx_model_path

# Step 2: ONNX -> TensorFlow
def convert_onnx_to_tf(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)
    
    # 保存為 TensorFlow 的 SavedModel 格式
    saved_model_dir = "saved_model"
    tf_rep.export_graph(saved_model_dir)
    print(f"[Info] ONNX 模型已轉換為 TensorFlow SavedModel 並保存至: {saved_model_dir}")

    # 打印 TensorFlow 模型參數大小
    print(f"[Info] TensorFlow 模型參數大小: {get_tensorflow_model_size(saved_model_dir):.2f} MB")
    return saved_model_dir

# Step 3: TensorFlow SavedModel -> TFLite
def convert_to_tflite_with_flex(saved_model_dir, tflite_model_path):
    # 初始化 TFLite 轉換器
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # 啟用 Flex Delegate 支持
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # 支持內建操作
        tf.lite.OpsSet.SELECT_TF_OPS    # 支持 TensorFlow Flex 操作
    ]

    # 啟用優化（可選）
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 轉換為 TFLite 模型
    tflite_model = converter.convert()

    # 保存 TFLite 模型
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"[Info] 模型已轉換為 TFLite 並保存至: {tflite_model_path}")

    # 打印 TFLite 模型大小（文件大小）
    tflite_model_size = os.path.getsize(tflite_model_path) / (1024 ** 2)
    print(f"[Info] TFLite 模型大小（文件大小）: {tflite_model_size:.2f} MB")

# 執行完整流程
onnx_model_path = convert_to_onnx()  # PyTorch -> ONNX
saved_model_dir = convert_onnx_to_tf(onnx_model_path)  # ONNX -> TensorFlow SavedModel
tflite_model_path = "conformer_model_with_flex.tflite"
convert_to_tflite_with_flex(saved_model_dir, tflite_model_path)  # TensorFlow SavedModel -> TFLite
