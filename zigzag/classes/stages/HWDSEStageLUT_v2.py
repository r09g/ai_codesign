# Libraries
from zigzag.classes.stages.Stage import Stage
from zigzag.utils import pickle_deepcopy

from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.operational_array import OperationalArray
from zigzag.classes.hardware.architecture.memory_level import MemoryLevel
from zigzag.classes.hardware.architecture.operational_unit import Multiplier
from zigzag.classes.hardware.architecture.operational_array import MultiplierArray
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core
import pandas as pd
import math
import itertools

import logging
logger = logging.getLogger(__name__)

def init_mem_cost_lut(path):
    lut = {}
    df = pd.read_csv(path, header=None)
    for row_idx in range(df.shape[0]):
        # all columns preceding last 2 columns are memory configurations, last 2 columns are energy and latency
        lut[tuple(df.iloc[row_idx,0:-2])] = list(df.iloc[row_idx,-2:])
    return lut

class HWDSEStageLUT_v2(Stage):
    """
    This stage is initialized only once. Memory hierarchy, process nodes, and PE array config is searched within this stage.
    """
    def __init__(self, list_of_callables, *, accelerator, mem_hierarchies, pe_array_factors, nodes, compute_costs, **kwargs):
        """
        Initialization of self.workload.
        :param main_inputs: MainInputs, NOT copied
        """
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        # dict: (mem_hierarchy: [[[r_port, w_port, rw_port], operands, port_alloc, served_dims] * #stages, acc_dim_ratio])
        self.mem_hierarchies = mem_hierarchies
        self.pe_array_factors = pe_array_factors  # list of integers, pe array scaling factors (all dims)
        self.nodes = nodes  # list, specifies process nodes
        self.compute_costs = compute_costs  # dict, maps node to compute cost
        # initialize memory cost lut based on Cacti data
        self.mem_cost_lut = init_mem_cost_lut('./zigzag/inputs/mem_cost_lut.csv')  # dict type   


    def run(self):
        ###########################################################
        # Tune this for search space
        stage_size_factors = [2**x for x in [3,4,5,7,9,10,13,16,18,19,21,23,24]]
        bw_size_factors = [2**x for x in [3,4,5,6,7,8,9,10,11,12]]
        ###########################################################
        for node in self.nodes:   
            for mh in self.mem_hierarchies:
                # generate combinations of stage sizes
                # if 4 stages with possible sizes [16,32,64] then combinations are:
                # [16,16,16,16], [16,16,16,32], [16,16,16,64], ..., [16,16,32,16], ..., [64,64,64,64]
                for pe_array in self.pe_array_factors:
                    # The length subtractions are hardcoded since the PE size information is embedded in the memory hierarchy
                    for stage_size_array in itertools.combinations_with_replacement(stage_size_factors, len(self.mem_hierarchies[mh])-2):
                    # for stage_size_array in [[128*8, 16, 131072*8*16]]:
                        # manually add memory hierarchy information for DRAM
                        stage_size_array = list(stage_size_array)
                        stage_size_array.append(10000000000)
                        for bw_size_array in itertools.combinations_with_replacement(bw_size_factors, len(self.mem_hierarchies[mh])-2):
                        # for bw_size_array in [[8, 16, 128*16]]:
                            bw_size_array = list(bw_size_array)
                            # manually add memory hierarchy information for DRAM
                            bw_size_array.append(64)
                            cfg = [mh, pe_array, stage_size_array, bw_size_array]
                            # print("\n> Candidate Configuration: " + str(cfg))
                            # update accelerator, search LUT for mem config
                            updated_accelerator = self.update_hw(self.mem_hierarchies[mh], stage_size_array, bw_size_array, pe_array, node)
                            # check if memory config is valid, skip invalid ones
                            if(updated_accelerator is None):
                                continue
                            kwargs = pickle_deepcopy(self.kwargs)
                            kwargs["accelerator"] = updated_accelerator
                            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
                            # configuration might be invalid
                            try:
                                for cme, extra_info in sub_stage.run():
                                    cme.cfg = cfg
                                    yield cme, extra_info
                                    continue
                            except:
                                # logger.info("> FAILED")
                                continue  # in case of error, move on to next configuration
                            else:
                                print("\n> Candidate Configuration: " + str(cfg))
                                print("> SUCCEEDED")
                                continue

    def update_hw(self, mh, stage_size_array, bw_size_array, pe_array, node):
        """
        mh: a list of parameters for a particular memory hierarchy
        stage_size_array: list of memory sizes for each stage from combination generator
        bw_size_array: list of bw sizes for each stage ...
        pe_array: pe array scaling factor for all dimensions
        node: node size
        """
        memory_instances = []
        stage_num = 0
        for stage in mh[0:-1]:
            cost_lut = self.get_mem_cost_lut(node=node,
                                             size=stage_size_array[stage_num],
                                             bw=bw_size_array[stage_num],
                                             r_port=stage[0][0],
                                             w_port=stage[0][1],
                                             rw_port=stage[0][2])  # get energy + latency from LUT
            # Invalid memory configuration, skip 
            if(cost_lut is None):
                # logger.info("Invalid LUT Memory Configuration")
                return None
            # Create new memory instances
            memory_instances.append(MemoryInstance(
                name="mem_inst_" + str(stage_num), 
                size=stage_size_array[stage_num], 
                r_bw=bw_size_array[stage_num], w_bw=bw_size_array[stage_num], 
                r_cost=cost_lut[0], 
                w_cost=cost_lut[0], 
                area=1, 
                r_port=stage[0][0], w_port=stage[0][1], rw_port=stage[0][2], 
                latency=cost_lut[1]))
            stage_num += 1
        # logger.info("Valid HW configuration")
        # create accelerator with updated params
        # compute block
        multiplier_input_precision = [8, 8]
        multiplier_energy = self.compute_costs[node]
        multiplier_area = 1
        # compute dimensions
        dimensions = {}
        for i in range(len(mh[-1])):
            dimensions['D' + str(i+1)] = round(pe_array * mh[-1][i])
        multiplier = Multiplier(multiplier_input_precision, multiplier_energy, multiplier_area)
        multiplier_array_inst = MultiplierArray(multiplier, dimensions)
        # memory block
        memory_hierarchy_inst = MemoryHierarchy(operational_array=multiplier_array_inst)
        stage_num = 0
        for stage in mh[0:-1]:
            memory_hierarchy_inst.add_memory(memory_instance=memory_instances[stage_num],
                                             operands=stage[1],
                                             port_alloc=stage[2],
                                             served_dimensions=stage[3])
            stage_num += 1
        # construct core
        core = {Core(1, multiplier_array_inst, memory_hierarchy_inst)}
        accelerator = Accelerator("hwdse_acc", core, None)
        return accelerator   


    def get_mem_cost_lut(self, node, size, bw, r_port, w_port, rw_port):
        # shape should be (1, 2)
        key = tuple([node,size,bw,r_port,w_port,rw_port])
        try:
            result = self.mem_cost_lut[key]
        except:
            result = None
        else:
            # print("Memory HW Found")
            pass
        return result
        

