import onnx
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader
from onnxruntime.quantization import QuantType, QuantFormat
import os

class ActivationCalibrationDataReader(CalibrationDataReader):
    def __init__(self, input_name, min_val, max_val, step):
        """キャリブレーションデータリーダーの初期化
        
        Args:
            input_name (str): 入力テンソルの名前
            min_val (float): キャリブレーションの最小値
            max_val (float): キャリブレーションの最大値
            step (float): キャリブレーションのステップサイズ
        """
        self.input_name = input_name
        self.data = np.arange(min_val, max_val + step, step, dtype=np.float32)
        self.index = 0
        
    def get_next(self):
        if self.index >= len(self.data):
            return None
        input_data = {
            self.input_name: np.array([self.data[self.index]], dtype=np.float32)
        }
        self.index += 1
        return input_data

def quantize_model_with_range(model_path, output_path, min_val, max_val, step):
    """指定された範囲でモデルを量子化
    
    Args:
        model_path (str): 入力モデルのパス
        output_path (str): 出力モデルのパス
        min_val (float): キャリブレーションの最小値
        max_val (float): キャリブレーションの最大値
        step (float): キャリブレーションのステップサイズ
    """
    model = onnx.load(model_path)
    input_name = model.graph.input[0].name
    
    calibration_data_reader = ActivationCalibrationDataReader(
        input_name, min_val, max_val, step
    )
    
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8
    )
    
    return output_path

def evaluate_model(session, input_name, test_values):
    """モデルの評価を実行
    
    Args:
        session: ONNXランタイムセッション
        input_name (str): 入力テンソルの名前
        test_values (np.ndarray): テスト用の入力値
    
    Returns:
        list: 出力値のリスト
    """
    outputs = []
    for value in test_values:
        input_data = np.array([value], dtype=np.float32)
        output = session.run(None, {input_name: input_data})[0]
        outputs.append(output[0])
    return outputs

def compare_quantized_models(original_path, ranges):
    """異なる量子化範囲のモデルを比較
    
    Args:
        original_path (str): オリジナルモデルのパス
        ranges (list): 量子化範囲のリスト。各要素は(min_val, max_val, step)のタプル
    """
    import onnxruntime
    
    # オリジナルモデルのセッション作成
    original_session = onnxruntime.InferenceSession(original_path)
    input_name = original_session.get_inputs()[0].name
    
    # テストデータの生成（広い範囲の値をカバー）
    test_values = np.array([-150, -100, -50, -10, 0, 10, 50, 100, 150], dtype=np.float32)
    
    # オリジナルモデルの出力を取得
    original_outputs = evaluate_model(original_session, input_name, test_values)
    
    print("\nModel Comparison Results:")
    print("-" * 100)
    print(f"{'Range':^30} | {'Size (KB)':^10} | {'Avg Error':^12} | {'Max Error':^12} | {'Size Reduction':^15}")
    print("-" * 100)
    
    original_size = os.path.getsize(original_path) / 1024
    
    for min_val, max_val, step in ranges:
        # 量子化モデルの生成
        range_str = f"{min_val:.1f} to {max_val:.1f} (step {step:.1f})"
        quantized_path = f"silu_model_int8_{min_val:.0f}_{max_val:.0f}_{step:.1f}.onnx"
        
        quantize_model_with_range(
            original_path, quantized_path, min_val, max_val, step
        )
        
        # 量子化モデルの評価
        quantized_session = onnxruntime.InferenceSession(quantized_path)
        quantized_outputs = evaluate_model(quantized_session, input_name, test_values)
        
        # エラーの計算
        errors = np.abs(np.array(original_outputs) - np.array(quantized_outputs))
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        # サイズの計算
        quantized_size = os.path.getsize(quantized_path) / 1024
        size_reduction = (original_size - quantized_size) / original_size * 100
        
        print(f"{range_str:<30} | {quantized_size:>10.2f} | {avg_error:>12.4f} | {max_error:>12.4f} | {size_reduction:>14.1f}%")

if __name__ == "__main__":
    # テストする量子化範囲のリスト
    # 各タプルは (min_val, max_val, step)
    calibration_ranges = [
        (-127, 128, 1),    # オリジナルの範囲
        (-256, 256, 2),    # より広い範囲
        (-64, 64, 0.5),    # より狭い範囲、細かいステップ
        (-512, 512, 4),    # さらに広い範囲、粗いステップ
        (-32, 32, 0.25),   # さらに狭い範囲、より細かいステップ
        (-8, 8, 0.0625),   # さらに狭い範囲、より細かいステップ

    ]
    
    silu_model = "silu_model.onnx"
    print("\nGenerating and comparing quantized models with different calibration ranges...")
    compare_quantized_models(silu_model, calibration_ranges)
