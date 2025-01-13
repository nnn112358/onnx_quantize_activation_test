import onnx
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader
from onnxruntime.quantization import QuantType, QuantFormat

class ActivationCalibrationDataReader(CalibrationDataReader):
    def __init__(self, input_name):
        """キャリブレーションデータリーダーの初期化"""
        self.input_name = input_name
        # -127から128までの値を1刻みで生成
        self.data = np.arange(-127, 129, 1, dtype=np.float32)
        self.index = 0
        
    def get_next(self):
        """次のキャリブレーションデータを取得"""
        if self.index >= len(self.data):
            return None
        input_data = {
            self.input_name: np.array([self.data[self.index]], dtype=np.float32)
        }
        self.index += 1
        return input_data

def quantize_model(model_path, output_path):
    """モデルをINT8に静的量子化
    
    Args:
        model_path (str): 入力モデルのパス
        output_path (str): 出力モデルのパス
    """
    # モデルの入力名を取得
    model = onnx.load(model_path)
    input_name = model.graph.input[0].name
    
    # キャリブレーションデータリーダーを作成
    calibration_data_reader = ActivationCalibrationDataReader(input_name)
    
    # 静的量子化を実行
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantFormat.QDQ,  # QDQフォーマットを使用
        per_channel=False,
        weight_type=QuantType.QInt8
    )
    
    return output_path

def test_quantized_model(original_path, quantized_path):
    """オリジナルモデルと量子化モデルを比較テスト
    
    Args:
        original_path (str): オリジナルモデルのパス
        quantized_path (str): 量子化モデルのパス
    """
    import onnxruntime
    
    # セッションの作成
    original_session = onnxruntime.InferenceSession(original_path)
    quantized_session = onnxruntime.InferenceSession(quantized_path)
    
    # 入力名の取得
    input_name = original_session.get_inputs()[0].name
    
    # テストデータの生成（-127から128までの代表的な値）
    test_values = np.array([-127, -64, -32, 0, 32, 64, 128], dtype=np.float32)
    
    print(f"\nComparing {original_path} vs {quantized_path}")
    print("Input\tOriginal\tQuantized\tDiff")
    print("-" * 50)
    
    for value in test_values:
        # 入力データの準備
        input_data = np.array([value], dtype=np.float32)
        
        # 推論実行
        original_output = original_session.run(None, {input_name: input_data})[0]
        quantized_output = quantized_session.run(None, {input_name: input_data})[0]
        
        # 結果の比較
        diff = abs(original_output - quantized_output)[0]
        
        print(f"{value:>6.1f}\t{original_output[0]:>8.4f}\t{quantized_output[0]:>8.4f}\t{diff:>8.4f}")

def compare_model_sizes(original_path, quantized_path):
    """モデルサイズを比較
    
    Args:
        original_path (str): オリジナルモデルのパス
        quantized_path (str): 量子化モデルのパス
    """
    import os
    
    original_size = os.path.getsize(original_path) / 1024  # KB
    quantized_size = os.path.getsize(quantized_path) / 1024  # KB
    
    print(f"\nModel size comparison:")
    print(f"Original model: {original_size:.2f} KB")
    print(f"Quantized model: {quantized_size:.2f} KB")
    print(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")

if __name__ == "__main__":
    # モデルのパス
    relu_model = "relu_model.onnx"
    relu_quantized = "relu_model_int8.onnx"
    silu_model = "silu_model.onnx"
    silu_quantized = "silu_model_int8.onnx"
    
    # ReLUモデルの量子化
    print("\nQuantizing ReLU model...")
    quantize_model(relu_model, relu_quantized)
    print("ReLU model quantization completed")
    
    # SiLUモデルの量子化
    print("\nQuantizing SiLU model...")
    quantize_model(silu_model, silu_quantized)
    print("SiLU model quantization completed")
    
    # モデルのテストと比較
    test_quantized_model(relu_model, relu_quantized)
    test_quantized_model(silu_model, silu_quantized)
    
    # モデルサイズの比較
    print("\nReLU models:")
    compare_model_sizes(relu_model, relu_quantized)
    print("\nSiLU models:")
    compare_model_sizes(silu_model, silu_quantized)