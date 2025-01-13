# onnx_quantize_activation_test

## the purpose

I investigated how the RELU and SILU of activation change with onnx's int8 quantization. It is said that RELU is advantageous for quantization and SILU has lower accuracy.

## Result

<img width="651" alt="image" src="https://github.com/user-attachments/assets/3522f643-46a2-45ce-91ac-a97ee7730a64" />


```
python pip install onnx onnxconverter-common
```

```
python generate-models.py
python onnx-fp16-converter.py
python inference-script.py
```

```
├── relu_model.onnx
├── relu_model_int8.onnx
├── silu_model.onnx
├── silu_model_fp16_converted.onnx
├── silu_model_int8.onnx
```





