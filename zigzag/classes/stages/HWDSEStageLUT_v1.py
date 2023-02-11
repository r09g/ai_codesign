# Libraries
from zigzag.classes.stages.Stage import Stage
from zigzag.utils import pickle_deepcopy

from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.operational_array import OperationalArray
import pandas as pd
import itertools

import logging
logger = logging.getLogger(__name__)

def init_mem_cost_lut(path):
    lut = {}
    df = pd.read_csv(path)
    for row_idx in range(df.shape[0]):
        # all columns preceding last 2 columns are memory configurations, last 2 columns are energy and latency
        lut[tuple(df.iloc[row_idx,0:-2])] = df.iloc[row_idx,-2:]
    return lut

class HWDSEStageLUT_v1(Stage):
    """
    This stage is initialized only once. Memory hierarchy, process nodes, and PE array config is searched within this stage.
    """
    def __init__(self, list_of_callables, *, accelerator, mem_hierarchies, pe_array_sizes, nodes, compute_costs, **kwargs):
        """
        Initialization of self.workload.
        :param main_inputs: MainInputs, NOT copied
        """
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        # dict: (mem_hierarchy: [[[r_port, w_port, rw_port]: List[int], operands: tuple(int), served_dims]] * #stages)
        self.mem_hierarchies = mem_hierarchies
        self.pe_array_sizes = pe_array_sizes  # list of integers, specifies pe array sizes
        self.nodes = nodes  # list, specifies process nodes
        self.compute_costs = compute_costs  # dict, maps node to compute cost
        
        # initialize memory cost lut based on Cacti data
        self.mem_cost_lut = init_mem_cost_lut('./inputs/mem_cost_lut.csv')  # dict type     


    def run(self):
        ###########################################################
        # Tune this for search space
        stage_size_factors = [2**x for x in range(6,12)]
        bw_size_factors = [2**x for x in range(6,7)]
        ###########################################################
        for node in self.nodes:   
            for mh in self.mem_hierarchies:
                # generate combinations of stage sizes
                # if 4 stages with possible sizes [16,32,64] then combinations are:
                # [16,16,16,16], [16,16,16,32], [16,16,16,64], ...
                for pe_array in self.pe_array_sizes:
                    for stage_size_array in itertools.combinations_with_replacement(stage_size_factors, len(self.mem_hierarchies[mh])):
                        for bw_size_array in itertools.combinations_with_replacement(bw_size_factors, len(self.mem_hierarchies[mh])):
                            updated_accelerator = self.update_hw(self.mem_hierarchies[mh], stage_size_array, bw_size_array, pe_array, node)
                            # check if memory config is valid, skip invalid ones
                            if(updated_accelerator is None):
                                pass
                            kwargs = pickle_deepcopy(self.kwargs)
                            kwargs["accelerator"] = updated_accelerator
                            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
                            # configuration might be invalid
                            try:
                                for cme, extra_info in sub_stage.run():
                                    cfg = [mh, pe_array, stage_size_array, bw_size_array]
                                    yield cme, cfg
                            except:
                                # print("Invalid HW Configuration")
                                pass  # in case of error, move on to next configuration
                                
                            
        

    def update_hw(self, mh, stage_size_array, bw_size_array, pe_array, node):
        accelerator_copy = pickle_deepcopy(self.accelerator)
        memory_instances = []
        stage_num = 0
        for stage in mh:
            cost_lut = self.get_mem_cost_lut(node=node,
                                             size=stage_size_array[stage_num],
                                             bw=bw_size_array[stage_num],
                                             r_port=stage[0][0],
                                             w_port=stage[0][1],
                                             rw_port=stage[0][2])  # get energy + latency from LUT
            # Invalid memory configuration, skip 
            if(cost_lut is None):
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
        
        for core in accelerator_copy.cores:
            # update compute cost
            operational_unit = core.operational_array.unit
            operational_unit.cost = self.compute_costs[node]
            # update operational array
            base_dims = core.operational_array.dimensions
            dimensions = {}
            for dim_obj in base_dims:
                if(dim_obj.name == 'D1'):
                    dimensions[dim_obj.name] = pe_array
                else:
                    dimensions[dim_obj.name] = 1
            operational_array = OperationalArray(operational_unit, dimensions)
            core.operational_array = operational_array
            # update memory hierarchy
            memory_hierarchy_graph = MemoryHierarchy(operational_array)
            stage_num = 0
            for stage in mh:
                memory_hierarchy_graph.add_memory(memory_instance=memory_instances[stage_num],
                                                  operands=stage[1],
                                                  served_dimensions=stage[2])
                stage_num += 1
            core.memory_hierarchy = memory_hierarchy_graph
            core.check_valid()
            core.recalculate_memory_hierarchy_information()
        return accelerator_copy   


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
        

