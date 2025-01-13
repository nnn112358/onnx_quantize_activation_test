# onnx_quantize_activation_test


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





