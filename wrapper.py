import os


if __name__ == "__main__":
    arch = "TPU_like"
    cmd1 = "python main_onnx.py --model inputs/examples/workloads/alexnet_inferred.onnx "
    cmd2 = "--accelerator inputs.examples.hardware." + arch + " "
    cmd3 = "--mapping inputs.examples.mapping.alexnet_on_" + arch
    os.system(cmd1 + cmd2 + cmd3)




























