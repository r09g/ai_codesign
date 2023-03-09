import onnx
from onnx import shape_inference

model = onnx.load("squeezenet1.0-12.onnx")
inferred_model = shape_inference.infer_shapes(model)
onnx.save(inferred_model, "squeezenet.onnx")
