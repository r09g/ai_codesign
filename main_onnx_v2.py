from classes.stages import *
import argparse

# Get the onnx model, the mapping and accelerator arguments
parser = argparse.ArgumentParser(description="Setup zigzag inputs")
parser.add_argument('--model', metavar='path', required=True, help='path to onnx model, e.g. inputs/examples/my_onnx_model.onnx')
parser.add_argument('--mapping', metavar='path', required=True, help='path to mapping file, e.g., inputs.examples.my_mapping')
parser.add_argument('--accelerator', metavar='path', required=True, help='module path to the accelerator, e.g. inputs.examples.accelerator1')
parser.add_argument('--arch', metavar='path', required=True, help='use name from --accelerator')
parser.add_argument('--node', metavar='path', required=True, help='process node in um, e.g. 0.090')
parser.add_argument('--compute_cost', metavar='path', required=True, help='compute cost in nJ, e.g. 0.003')
args = parser.parse_args()

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
    MinimalEnergyStage,
    HWIteratorStageCacti_v2,  # Example stage that varies the rf energy scaling
    SumStage,  # Adds all CME of all the layers together, getting the total energy, latency, ...
    WorkloadStage,  # Iterates through the different layers in the workload
    SpatialMappingConversionStage,  # Generates multiple spatial mappings (SM)
    MinimalEnergyStage,  # Reduces all CMEs, returning minimal latency one
    LomaStage,  # Generates multiple temporal mappings (TM)
    CostModelStage  # Evaluates generated SM and TM through cost model
],
    accelerator_path=args.accelerator,  # required by AcceleratorParserStage
    onnx_model_path=args.model,  # required by ONNXModelParserStage
    mapping_path=args.mapping,  # required by ONNXModelParserStage
    dump_filename_pattern="outputs/{datetime}.json",  # output file save pattern
    loma_lpf_limit=6,  # required by LomaStage
    iterator_arch=args.arch,
    node=args.node,
    compute_cost=args.compute_cost
)

# Launch the MainStage
mainstage.run()
