import onnx
from onnxconverter_common import float16
import numpy as np
import onnxruntime

def convert_float_to_float16(model_path, output_path):
    """
    ONNXモデルをFP16に変換
    Args:
        model_path: 入力モデルのパス
        output_path: 出力モデルのパス
    """
    # モデルを読み込む
    model = onnx.load(model_path)
    
    # モデルをFP16に変換
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,  # 入出力の型をfloat32のまま保持
        op_block_list=None,  # 変換から除外する操作がある場合はここで指定
        node_block_list=None  # 変換から除外するノードがある場合はここで指定
    )
    
    # 変換したモデルを保存
    onnx.save(model_fp16, output_path)
    
    return output_path

def test_model(model_path):
    """
    変換したモデルをテストする
    """
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # テストデータ
    test_inputs = np.array([-127.0, -64.0, 0.0, 64.0, 128.0], dtype=np.float32)
    
    print(f"\nTesting model: {model_path}")
    print("Input\tOutput")
    print("-" * 30)
    
    for x in test_inputs:
        input_data = np.array([x], dtype=np.float32)
        output = session.run(None, {input_name: input_data})[0]
        print(f"{x:>6.1f}\t{output[0]:>10.6f}")

def compare_model_sizes(original_path, fp16_path):
    """
    元のモデルとFP16モデルのサイズを比較
    """
    import os
    original_size = os.path.getsize(original_path) / 1024  # KB
    fp16_size = os.path.getsize(fp16_path) / 1024  # KB
    
    print("\nModel size comparison:")
    print(f"Original model: {original_size:.2f} KB")
    print(f"FP16 model: {fp16_size:.2f} KB")
    print(f"Size reduction: {((original_size - fp16_size) / original_size * 100):.1f}%")

def verify_model(model_path):
    """
    モデルの検証を行う
    """
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    
    # モデルの情報を表示
    print("\nModel information:")
    for input_info in model.graph.input:
        print(f"Input: {input_info.name}, Type: {input_info.type.tensor_type.elem_type}")
    for output_info in model.graph.output:
        print(f"Output: {output_info.name}, Type: {output_info.type.tensor_type.elem_type}")

if __name__ == "__main__":
    # 入力と出力のパス
    input_model = "silu_model.onnx"
    output_model = "silu_model_fp16_converted.onnx"
    
    # FP16変換
    print("Converting model to FP16...")
    converted_model_path = convert_float_to_float16(input_model, output_model)
    print(f"Converted model saved to: {converted_model_path}")
    
    # モデルの検証
    print("\nVerifying converted model...")
    verify_model(converted_model_path)
    
    # サイズ比較
    compare_model_sizes(input_model, converted_model_path)
    
    # モデルのテスト
    print("\nTesting both models for comparison:")
    print("\nOriginal model:")
    test_model(input_model)
    print("\nFP16 converted model:")
    test_model(converted_model_path)
