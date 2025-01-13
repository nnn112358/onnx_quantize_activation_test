import onnx
import numpy as np
from onnx import helper, TensorProto, numpy_helper

def create_relu_model(output_path="relu_model.onnx"):
    """
    Create an ONNX model that applies ReLU activation to scalar input
    
    Args:
        output_path (str): Path to save the ONNX model
    """
    # Create input (ValueInfoProto)
    X = helper.make_tensor_value_info(
        name='input',  
        elem_type=TensorProto.FLOAT,
        shape=[1]  # scalar input
    )
    
    # Create output (ValueInfoProto) 
    Y = helper.make_tensor_value_info(
        name='output',
        elem_type=TensorProto.FLOAT,
        shape=[1]  # scalar output
    )
    
    # Create ReLU node
    relu = helper.make_node(
        'Relu',
        inputs=['input'],
        outputs=['output']
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[relu],
        name='relu_model',
        inputs=[X],
        outputs=[Y],
        initializer=[]
    )
    
    # Create the model
    model = helper.make_model(
        graph,
        producer_name='relu_model_generator',
        opset_imports=[helper.make_opsetid("", 13)]
    )
    
    # Verify the model
    onnx.checker.check_model(model)
    
    # Save the model
    onnx.save(model, output_path)
    
    return model

def create_silu_model(output_path="silu_model.onnx"):
    """
    Create an ONNX model that applies SiLU activation to scalar input
    
    Args:
        output_path (str): Path to save the ONNX model
    """
    # Create input (ValueInfoProto)
    X = helper.make_tensor_value_info(
        name='input',  
        elem_type=TensorProto.FLOAT,
        shape=[1]  # scalar input
    )
    
    # Create output (ValueInfoProto) 
    Y = helper.make_tensor_value_info(
        name='output',
        elem_type=TensorProto.FLOAT,
        shape=[1]  # scalar output
    )
    
    # Create Sigmoid node
    sigmoid = helper.make_node(
        'Sigmoid',
        inputs=['input'],
        outputs=['sigmoid_output']
    )
    
    # Create Mul node (multiply input with sigmoid output)
    mul = helper.make_node(
        'Mul',
        inputs=['input', 'sigmoid_output'],
        outputs=['output']
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[sigmoid, mul],
        name='silu_model',
        inputs=[X],
        outputs=[Y],
        initializer=[]
    )
    
    # Create the model
    model = helper.make_model(
        graph,
        producer_name='silu_model_generator',
        opset_imports=[helper.make_opsetid("", 13)]
    )
    
    # Verify the model
    onnx.checker.check_model(model)
    
    # Save the model
    onnx.save(model, output_path)
    
    return model

def test_model(model_path, test_input):
    """
    Test the ONNX model with given input
    
    Args:
        model_path (str): Path to the ONNX model
        test_input (float): Input value to test
    """
    # Create input array
    x = np.array([test_input], dtype=np.float32)
    
    # Run ONNX model
    session = onnxruntime.InferenceSession(model_path)
    onnx_output = session.run(None, {'input': x})[0]
    
    # Calculate expected output manually
    if "relu" in model_path:
        expected_output = np.maximum(0, x)
    else:  # silu
        expected_output = x * (1 / (1 + np.exp(-x)))
    
    # Check if outputs are close
    np.testing.assert_allclose(onnx_output, expected_output, rtol=1e-5)
    
    print(f"\nTesting {model_path}:")
    print(f"Input: {x}")
    print(f"Model output: {onnx_output}")
    print(f"Expected output: {expected_output}")
    print("Test passed!")

if __name__ == "__main__":
    import onnxruntime
    
    # Create ReLU model
    relu_model = create_relu_model()
    print("ReLU model created and saved successfully!")
    
    # Create SiLU model
    silu_model = create_silu_model()
    print("SiLU model created and saved successfully!")
    
    # Test both models with different inputs
    test_values = [-2.0, 0.0, 2.0]
    
    for test_value in test_values:
        test_model("relu_model.onnx", test_value)
        test_model("silu_model.onnx", test_value)
