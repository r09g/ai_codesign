import networkx as nx

from classes.stages.Stage import Stage
from classes.workload.dnn_workload import DNNWorkload
import classes.io.input_config as inputs
from classes.workload.dummy_layer_node import DummyNode
from utils import pickle_deepcopy
from networkx import DiGraph

import pandas as pd
import math
import pickle
from classes.hardware.architecture.memory_instance import MemoryInstance
from classes.hardware.architecture.memory_hierarchy import MemoryHierarchy

import logging
logger = logging.getLogger(__name__)


class HWIteratorStage(Stage):
    """
    Example stage that scales the energy cost of reading and writing a RF.
    For each HW design point we query the underlying stages and get back the optimal latency solution.
    In this example, I set the class (MemoryLevel) attributes directly. In practice, this should be done with a setter function.
    """
    def __init__(self, list_of_callables, *, accelerator, **kwargs):
        """
        Initialization of self.workload.
        :param main_inputs: MainInputs, NOT copied
        """
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator  # This is the accelerator object that contains 
        self.rf_capacity_list = [x*8 for x in list(range(8,512+8,8))] 
        self.sram_capacity_list = [x*8*1024 for x in [256]]
        self.dram_capacity_list = [10000000000]
        

    def run(self):
        #TODO
        design_data_point = []
        for rf_w in self.rf_capacity_list:
            for rf_o in [512*8]:
                for sram_size in self.sram_capacity_list:
                    for dram_size in self.dram_capacity_list:
                        # MODIFICATION OF THE ACCELERATOR OBJECT
                        updated_accelerator = self.update_accelerator_memory(self.accelerator,[rf_w, rf_o, sram_size, dram_size])
                        # QUERY THE UNDERLYING STAGES WITH THE UPDATED ACCELERATOR
                        # We copy the kwargs just to be sure we don't accidentally keep unwanted changes
                        kwargs = pickle_deepcopy(self.kwargs)
                        kwargs["accelerator"] = updated_accelerator
                        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
                        for cme, extra_info in sub_stage.run():
                            energy_total = cme.energy_total
                            logger.info(f"Total network energy for configuration {rf_w}:{rf_o}:{sram_size}:{dram_size} = {energy_total:.3e}")
                            design_data_point.append([rf_w, rf_o, sram_size, dram_size, cme.mem_energy])
                            yield cme, extra_info
                            # sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        with open('design_data_point.pkl', 'wb') as file:
            pickle.dump(design_data_point, file)

    @staticmethod
    def update_accelerator_memory(accelerator, mem_config):
        accelerator_copy = pickle_deepcopy(accelerator)
        # construct memory
        rf_w_inst = MemoryInstance(name="rf_w", size=mem_config[0], r_bw=8, w_bw=8, r_cost=0.095*math.sqrt(mem_config[0]), w_cost=0.095*math.sqrt(mem_config[0]), area=0,
                                     r_port=1, w_port=1, rw_port=0, latency=1)
        rf_o_inst = MemoryInstance(name="rf_o", size=mem_config[1], r_bw=8, w_bw=8, r_cost=0.095*math.sqrt(mem_config[1]), w_cost=0.095*math.sqrt(mem_config[1]), area=0,
                                     r_port=2, w_port=2, rw_port=0, latency=1)
        # rf_o_inst = MemoryInstance(name="rf_o", size=16, r_bw=16, w_bw=16, r_cost=0.54, w_cost=0.6, area=0,
        #                           r_port=2, w_port=2, rw_port=0, latency=1)
        sram_inst = \
        MemoryInstance(name="sram", size=mem_config[2], r_bw=128*16, w_bw=128*16, r_cost=26.01*math.sqrt(mem_config[2])*16, w_cost=23.65*math.sqrt(mem_config[2])*16, area=0,
                       r_port=1, w_port=1, rw_port=0, latency=1, min_r_granularity=64, min_w_granularity=64)
        dram_inst = MemoryInstance(name="dram", size=10000000000, r_bw=64, w_bw=64, r_cost=700, w_cost=750, area=0,
                              r_port=0, w_port=0, rw_port=1, latency=1)
        # As our accelerator can have different cores, iterate through all its cores
        for core in accelerator_copy.cores:
            # Construct new memory hierarchy
            memory_hierarchy_graph = MemoryHierarchy(core.memory_hierarchy.operational_array)
            memory_hierarchy_graph.add_memory(memory_instance=rf_w_inst, operands=('I2',),
                                              port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                              served_dimensions={(0, 0)})
            memory_hierarchy_graph.add_memory(memory_instance=rf_o_inst, operands=('O',),
                                              port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_2', 'th': 'r_port_2'},),
                                              served_dimensions={(0, 1)})
            memory_hierarchy_graph.add_memory(memory_instance=sram_inst, operands=('I1', 'O'),
                                              port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},
                                                          {'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_1', 'th': 'r_port_1'},),
                                              served_dimensions='all')
            memory_hierarchy_graph.add_memory(memory_instance=dram_inst, operands=('I1', 'I2', 'O'),
                                              port_alloc=({'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},
                                                          {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},
                                                          {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': 'rw_port_1', 'th': 'rw_port_1'},),
                                              served_dimensions='all')
            core.memory_hierarchy = memory_hierarchy_graph
            core.check_valid()
            core.recalculate_memory_hierarchy_information()

        return accelerator_copy

