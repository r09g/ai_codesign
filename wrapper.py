import os


if __name__ == "__main__":
    cmd1 = "python main_onnx.py --model zigzag/inputs/examples/workload/alexnet.onnx "
    cmd2 = "--accelerator zigzag.inputs.examples.hardware.TPU_like "
    cmd3 = "--mapping zigzag.inputs.examples.mapping.tpu_like"
    os.system(cmd1 + cmd2 + cmd3)




























