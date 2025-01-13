import onnx
import numpy as np
from onnx import helper, TensorProto
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat
from onnxruntime.quantization import QuantType

def create_relu_model(output_path="relu_model.onnx"):
    """ReLUのONNXモデルを作成"""
    # Input/Outputの定義
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None])
    
    # ReLUノードの作成
    node_def = helper.make_node(
        'Relu',
        inputs=['input'],
        outputs=['output'],
    )
    
    # グラフとモデルの作成
    graph_def = helper.make_graph(
        [node_def],
        'relu-model',
        [X],
        [Y],
    )
    
    model_def = helper.make_model(graph_def, producer_name='relu-model')
    model_def.opset_import[0].version = 13  # opsetバージョンを設定
    onnx.save(model_def, output_path)
    return output_path

def create_silu_model(output_path="silu_model.onnx"):
    """SiLU (Swish)のONNXモデルを作成"""
    # Input/Outputの定義
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None])
    
    # Sigmoidノードの作成
    sigmoid_node = helper.make_node(
        'Sigmoid',
        inputs=['input'],
        outputs=['sigmoid_output'],
    )
    
    # 乗算ノードの作成
    mul_node = helper.make_node(
        'Mul',
        inputs=['input', 'sigmoid_output'],
        outputs=['output'],
    )
    
    # グラフとモデルの作成
    graph_def = helper.make_graph(
        [sigmoid_node, mul_node],
        'silu-model',
        [X],
        [Y],
    )
    
    model_def = helper.make_model(graph_def, producer_name='silu-model')
    model_def.opset_import[0].version = 13  # opsetバージョンを設定
    onnx.save(model_def, output_path)
    return output_path

class CalibrationReader(CalibrationDataReader):
    def __init__(self, input_name):
        self.input_name = input_name
        self.data = np.arange(-127.0, 129.0, 1.0).astype(np.float32)
        self.index = 0
        
    def get_next(self):
        if self.index >= len(self.data):
            return None
        input_data = {self.input_name: np.array([self.data[self.index]], dtype=np.float32)}
        self.index += 1
        return input_data

def quantize_model(model_path, output_path):
    """モデルをInt8に量子化"""
    # モデルの入力名を取得
    model = onnx.load(model_path)
    input_name = model.graph.input[0].name
    
    # キャリブレーションデータリーダーを作成
    calibration_data_reader = CalibrationReader(input_name)
    
    # 量子化の実行
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantFormat.QDQ,  # QDQフォーマットを使用
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8
    )
    return output_path

def test_model(model_path, activation_name=""):
    """モデルをテスト"""
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # テストデータ
    test_inputs = np.array([-127.0, -64.0, -32.0, 0.0, 32.0, 64.0, 128.0], dtype=np.float32)
    
    print(f"\nTesting {activation_name} model: {model_path}")
    print("Input\tOutput")
    print("-" * 30)
    
    for x in test_inputs:
        input_data = np.array([x], dtype=np.float32)
        output = session.run(None, {input_name: input_data})[0]
        print(f"{x:>4.0f}\t{output[0]:>10.6f}")
        
        # 期待値の計算（比較用）
        if "relu" in model_path.lower():
            expected = max(0, x)
        else:  # SiLU
            expected = x * (1 / (1 + np.exp(-x)))
        print(f"      \t{expected:>10.6f} (expected)")
        print("-" * 30)

if __name__ == "__main__":
    # ReLUモデルの作成と量子化
    relu_model = create_relu_model()
    print(f"Created ReLU model: {relu_model}")
    relu_quantized = quantize_model(relu_model, "relu_model_int8.onnx")
    print(f"Created quantized ReLU model: {relu_quantized}")
    
    # SiLUモデルの作成と量子化
    silu_model = create_silu_model()
    print(f"Created SiLU model: {silu_model}")
    silu_quantized = quantize_model(silu_model, "silu_model_int8.onnx")
    print(f"Created quantized SiLU model: {silu_quantized}")
    
    # すべてのモデルのテスト
    test_model(relu_model, "ReLU")
    test_model(relu_quantized, "Quantized ReLU")
    test_model(silu_model, "SiLU")
    test_model(silu_quantized, "Quantized SiLU")