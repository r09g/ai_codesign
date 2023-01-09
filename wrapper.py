import os

def run(workload):  # only supports alexnet as of now until mapping can be automated
    cmd1 = "python main_onnx_v2.py --model " + str(workload) + " "
    arch_list = ["TPU_like","Ascend_like","Edge_TPU_like","Eyeriss_like","Meta_prototype","Tesla_NPU_like"]
    node_list = [0.090,0.065,0.045,0.032]  # in um
    compute_cost_table = [0.0016,0.0006,0.0002,0.00007192]  # in nJ for 8-bit multiplication
    # assume delay decreases linearly and power decreases quadratically with channel length L
    node_cost = dict(zip(node_list, compute_cost_table))

    # go over different architectures
    for arch in arch_list:
        cmd2 = "--accelerator inputs.examples.hardware." + arch + " --mapping inputs.examples.mapping.alexnet_on_" + arch + " --arch " + arch
        for node in node_list:
            cmd3 = " --node " + str(node) + " --compute_cost " + str(node_cost[node])

            os.system(cmd1 + cmd2 + cmd3)


if __name__ == "__main__":
    run("inputs/examples/workloads/alexnet_inferred.onnx")





























