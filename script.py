import os
import torch
import torch.onnx
from conformer.conformer.encoder import ConformerEncoder

class WrapperModel(torch.nn.Module):
    def __init__(self, conformer_encoder):
        super(WrapperModel, self).__init__()
        self.conformer_encoder = conformer_encoder

    def forward(self, inputs):
        # 自動計算 input_lengths
        input_lengths = torch.tensor([inputs.shape[1]] * inputs.shape[0], dtype=torch.long, device=inputs.device)
        outputs, _ = self.conformer_encoder(inputs, input_lengths)
        return outputs

def get_model_size_info(pt_file_path, model_class):
    if not os.path.exists(pt_file_path):
        raise FileNotFoundError(f"文件 {pt_file_path} 不存在")

    checkpoint = torch.load(pt_file_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = None

    if state_dict:
        model = model_class()
        model.load_state_dict(state_dict)
    else:
        model = checkpoint

    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 ** 2)
    file_size_mb = os.path.getsize(pt_file_path) / (1024 ** 2)

    return {
        "total_params": total_params,
        "file_size_mb": file_size_mb,
        "theoretical_size_mb": model_size_mb
    }, model

def export_to_onnx(model, dummy_input, onnx_path, opset_version=11):
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size", 1: "sequence_length"}, "output": {0: "batch_size"}}
    )
    print(f"ONNX 模型已保存至: {onnx_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check PyTorch model size and export to ONNX")
    parser.add_argument("--pt_file", type=str, required=True, help="Path to the .pt file")
    parser.add_argument("--onnx_path", type=str, default="model.onnx", help="Path to save the .onnx file")
    parser.add_argument("--input_dim", type=int, default=40, help="Input feature dimension for the model")
    parser.add_argument("--encoder_dim", type=int, default=256, help="Encoder dimension for the model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the encoder")
    parser.add_argument("--num_attention_heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--seq_length", type=int, default=100, help="Sequence length for the dummy input")
    parser.add_argument("--opset_version", type=int, default=11, help="ONNX opset version")
    args = parser.parse_args()

    model_info, conformer_model = get_model_size_info(
        args.pt_file,
        lambda: ConformerEncoder(
            input_dim=args.input_dim,
            encoder_dim=args.encoder_dim,
            num_layers=args.num_layers,
            num_attention_heads=args.num_attention_heads
        )
    )
    print(f"模型參數總數: {model_info['total_params']}")
    print(f"模型文件大小: {model_info['file_size_mb']:.2f} MB")
    print(f"模型理論大小: {model_info['theoretical_size_mb']:.2f} MB")

    wrapped_model = WrapperModel(conformer_model)

    dummy_input = torch.randn(1, args.seq_length, args.input_dim)
    export_to_onnx(wrapped_model, dummy_input, args.onnx_path, opset_version=args.opset_version)
