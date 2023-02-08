from classes.stages import *

# Get the onnx model, the mapping and accelerator arguments
mem_hierarchies = {'TPU_like': [[[1,1,0],('I2'),{(0, 0)}],
                                [[2,2,0],('O'),{(0, 1)}],
                                [[1,1,0],('I1', 'O'),'all'],
                                [[0,0,1],('I1', 'I2', 'O'),'all']]}

pe_array_sizes = [4,16]
nodes = ['SRAM65']
compute_costs = {'SRAM65': 1}  # arbitrary for now

# Initialize the logger
import logging as _logging
_logging_level = _logging.INFO
# _logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
_logging_format = '%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
_logging.basicConfig(level=_logging_level,
                     format=_logging_format)

# Initialize the MainStage which will start execution.
# The first argument of this init is the list of stages that will be executed in sequence.
# The second argument of this init are the arguments required for these different stages.
mainstage = MainStage([  # Initializes the MainStage as entry point
    ONNXModelParserStage,  # Parses the ONNX Model into the workload
    AcceleratorParserStage,  # Parses the accelerator
    SimpleSaveStage,
    # MinimalEnergyStage,
    HWDSEStageLUT_v1,  # Example stage that varies the rf energy scaling
    SumStage,  # Adds all CME of all the layers together, getting the total energy, latency, ...
    WorkloadStage,  # Iterates through the different layers in the workload
    MinimalEnergyStage,
    SpatialMappingGeneratorStage,  # Generates multiple spatial mappings (SM)
    MinimalEnergyStage,  # Reduces all CMEs, returning minimal latency one
    LomaStage,  # Generates multiple temporal mappings (TM)
    CostModelStage  # Evaluates generated SM and TM through cost model
],
    accelerator_path="inputs.examples.hardware.TPU_like",  # used as entry point
    onnx_model_path="inputs/examples/workloads/alexnet_inferred.onnx",  # required by ONNXModelParserStage
    mapping_path="inputs.examples.mapping.alexnet_on_TPU_like",  # required by ONNXModelParserStage
    dump_filename_pattern="outputs/{datetime}.json",  # output file save pattern
    loma_lpf_limit=6,  # required by LomaStage
    mem_hierarchies=mem_hierarchies,
    pe_array_sizes=pe_array_sizes,
    nodes=nodes,
    compute_costs=compute_costs
)

# Launch the MainStage
mainstage.run()

