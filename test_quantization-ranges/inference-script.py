import onnxruntime
import numpy as np
import pandas as pd

def run_inference(model_path):
    """指定されたモデルで推論を実行"""
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # 入力範囲の生成 (-127から128まで、1刻み)
    inputs = np.arange(-127, 129, 0.1, dtype=np.float32)
    outputs = []
    
    # 推論実行
    for x in inputs:
        input_data = np.array([x], dtype=np.float32)
        output = session.run(None, {input_name: input_data})[0]
        outputs.append(output[0])
    
    return inputs, np.array(outputs)

def save_results_to_csv(inputs, outputs_dict, output_file):
    """結果をCSVファイルに保存"""
    results = {
        'input': inputs
    }
    results.update(outputs_dict)
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    # モデルパス
    models = {
        'silu_model': 'silu_model.onnx',
        'silu_model_int8_-8_8': 'silu_model_int8_-8_8_0.1.onnx',
        'silu_model_int8_-32_32': 'silu_model_int8_-32_32_0.2.onnx',
        'silu_model_int8_-64_64': 'silu_model_int8_-64_64_0.5.onnx',
        'silu_model_int8_-127_128': 'silu_model_int8_-127_128_1.0.onnx',
        'silu_model_int8_-256_256': 'silu_model_int8_-256_256_2.0.onnx',
        'silu_model_int8_-512_512': 'silu_model_int8_-512_512_4.0.onnx',
    }
    
    # すべてのモデルで推論実行
    inputs = None
    outputs_dict = {}
    
    for name, model_path in models.items():
        print(f"Running inference on {name} model...")
        input_values, output_values = run_inference(model_path)
        
        if inputs is None:
            inputs = input_values
        
        outputs_dict[name] = output_values
    
    # 結果をCSVファイルに保存
    save_results_to_csv(inputs, outputs_dict, 'activation_results.csv')
    
    # データの確認出力
    print("\nSample of results:")
    df = pd.DataFrame({
        'input': inputs,
        **outputs_dict
    })
    print(df.head())
    print("\nCSV columns:", df.columns.tolist())
    print(f"Total rows: {len(df)}")

if __name__ == "__main__":
    main()
