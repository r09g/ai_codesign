from zigzag.classes.stages import *
import argparse
import re

# Get the onnx model, the mapping and accelerator arguments
mem_hierarchies = {'tesla_npu_like':
                    [[[1,1,0],('I2',),({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),{(0, 1, 0), (0, 0, 1)}],
                      [[2,2,0],('O',),({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_2', 'th': 'r_port_2'},),{(0, 0, 0)}],
                      [[1,1,0],('I1',),({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),'all'],
                      [[1,1,0],('I2',),({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),'all'],
                      [[1,1,0],('I2',),({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),'all'],
                      [[1,1,0],('I1', 'O'),({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},{'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_1', 'th': 'r_port_1'},),'all'],
                      [[0,0,1],('I1', 'I2', 'O',),({'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},{'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},{'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': 'rw_port_1', 'th': 'rw_port_1'},),'all'],
                      (1,1/4,1/8)]}

pe_array_factors = [16,32,64]
nodes = ['SRAM65']
compute_costs = {'SRAM65': 1}  # arbitrary for now

# Initialize the logger
import logging as _logging
_logging_level = _logging.INFO
_logging_format = '%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
_logging.basicConfig(level=_logging_level,
                     format=_logging_format)

args_accelerator = "zigzag.inputs.examples.hardware.TPU_like"
args_workload = "zigzag/inputs/examples/workload/alexnet.onnx"
args_mapping = "zigzag.inputs.examples.mapping.tpu_like"

hw_name = args_accelerator.split(".")[-1]
wl_name = re.split(r"/|\.", args_workload)[-1]
if wl_name == 'onnx':
    wl_name = re.split(r"/|\.", args_workload)[-2]
experiment_id = f"{hw_name}-{wl_name}"
pkl_name = f'{experiment_id}-saved_list_of_cmes'

# Initialize the MainStage which will start execution.
# The first argument of this init is the list of stages that will be executed in sequence.
# The second argument of this init are the arguments required for these different stages.
mainstage = MainStage([  # Initializes the MainStage as entry point
    ONNXModelParserStage,  # Parses the ONNX Model into the workload
    AcceleratorParserStage,  # Parses the accelerator
    csvStage,
    HWDSEStageLUT_v2,
    SumStage,
    WorkloadStage,  # Iterates through the different layers in the workload
    MinimalEnergyStage,
    SpatialMappingGeneratorStage,  # Generates multiple spatial mappings (SM)
    LomaStage,  # Generates multiple temporal mappings (TM)
    CostModelStage  # Evaluates generated SM and TM through cost model
],
    accelerator=args_accelerator,  # used as entry point
    workload=args_workload,  # required by ONNXModelParserStage
    mapping=args_mapping,  # required by ONNXModelParserStage
    dump_filename_pattern=f"outputs/{experiment_id}-layer_?.json",  # output file save pattern
    loma_lpf_limit=6,  # required by LomaStage
    loma_show_progress_bar=False,  # shows a progress bar while iterating over temporal mappings
    mem_hierarchies=mem_hierarchies,
    pe_array_factors=pe_array_factors,
    nodes=nodes,
    compute_costs=compute_costs
)

# Launch the MainStage
mainstage.run()
