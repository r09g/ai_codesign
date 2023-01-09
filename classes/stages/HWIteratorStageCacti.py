# Libraries
import networkx as nx

from classes.stages.Stage import Stage
from classes.workload.dnn_workload import DNNWorkload
import classes.io.input_config as inputs
from classes.workload.dummy_layer_node import DummyNode
from utils import pickle_deepcopy
from networkx import DiGraph

import os
import pickle
from classes.hardware.architecture.memory_instance import MemoryInstance
from classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from helper_scripts_rhyang.cacti import get_cacti

import logging
logger = logging.getLogger(__name__)

# Debug Options
CACTI_ON = True


class HWIteratorStageCacti(Stage):
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
        self.rf_o_size_bytes = [256,512]
        self.rf_o_bw_list = [8]
        self.rf_w_size_bytes = [256,512]
        self.rf_w_bw_list = [8] 
        self.sram_size_bytes = [24*1024]
        self.sram_bw_list = [128]

    def run(self):
        #TODO
        design_data_point = []
        for rf_w_size in self.rf_w_size_bytes:
            for rf_w_bw in self.rf_w_bw_list:
                for rf_o_size in self.rf_o_size_bytes:
                    for rf_o_bw in self.rf_o_bw_list:
                        for sram_size in self.sram_size_bytes:
                            for sram_bw in self. sram_bw_list:
                                # MODIFICATION OF THE ACCELERATOR OBJECT
                                updated_accelerator = self.update_accelerator_memory(self.accelerator,[rf_w_size, rf_w_bw, rf_o_size, rf_o_bw, sram_size, sram_bw])
                                # QUERY THE UNDERLYING STAGES WITH THE UPDATED ACCELERATOR
                                # We copy the kwargs just to be sure we don't accidentally keep unwanted changes
                                kwargs = pickle_deepcopy(self.kwargs)
                                kwargs["accelerator"] = updated_accelerator
                                sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
                                for cme, extra_info in sub_stage.run():
                                    energy_total = cme.energy_total
                                    # logger.info(f"Total network energy for configuration {rf_w}:{rf_o}:{sram_size}:{dram_size} = {energy_total:.3e}")
                                    design_data_point.append([rf_w_size, rf_w_bw, rf_o_size, rf_o_bw, sram_size, sram_bw, cme.mem_energy])
                                    yield cme, extra_info
                                    # sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)

        if os.path.exists('design_data_point.pkl'):
            os.remove('design_data_point.pkl')
        with open('./design_data_point.pkl', 'wb') as file:
            pickle.dump(design_data_point, file)

    @staticmethod
    def update_accelerator_memory(accelerator, mem_config):
        accelerator_copy = pickle_deepcopy(accelerator)
        if(CACTI_ON):
            rf_w_data = get_cacti(mem_config[0], mem_config[1], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            rf_o_data = get_cacti(mem_config[2], mem_config[3], 2, '/rsgs/scratch0/rhyang/codesign/cacti')
            sram_data = get_cacti(mem_config[4], mem_config[5], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
        else:
            rf_w_data = {}
            rf_w_data['node'] = 90
            rf_w_data['size_bytes'] = float(mem_config[0])
            rf_w_data['bw'] = float(mem_config[1])
            rf_w_data['latency_ns'] = 1
            rf_w_data['cost'] = 1
            rf_w_data['port'] = 1
            rf_o_data = {}
            rf_o_data['node'] = 90
            rf_o_data['size_bytes'] = float(mem_config[2])
            rf_o_data['bw'] = float(mem_config[3])
            rf_o_data['latency_ns'] = 1
            rf_o_data['cost'] = 1
            rf_o_data['port'] = 2
            sram_data = {}
            sram_data['node'] = 90
            sram_data['size_bytes'] = float(mem_config[4])
            sram_data['bw'] = float(mem_config[5])
            sram_data['latency_ns'] = 1
            sram_data['cost'] = 1
            sram_data['port'] = 1

        # construct memory
        rf_w_inst = MemoryInstance(name="rf_w", size=(rf_w_data['size_bytes'])*8, r_bw=rf_w_data['bw'], w_bw=rf_w_data['bw'], r_cost=rf_w_data['cost'], w_cost=rf_w_data['cost'], area=0,
                                     r_port=1, w_port=1, rw_port=0, latency=rf_w_data['latency_ns'])
        rf_o_inst = MemoryInstance(name="rf_o", size=(rf_o_data['size_bytes'])*8, r_bw=2*(rf_o_data['bw']), w_bw=2*(rf_o_data['bw']), r_cost=rf_o_data['cost'], w_cost=rf_o_data['cost'], area=0,
                                     r_port=2, w_port=2, rw_port=0, latency=rf_o_data['latency_ns'])
        sram_inst = \
        MemoryInstance(name="sram", size=(sram_data['size_bytes'])*8, r_bw=sram_data['bw'], w_bw=sram_data['bw'], r_cost=sram_data['cost'], w_cost=sram_data['cost'], area=0,
                       r_port=1, w_port=1, rw_port=0, latency=sram_data['latency_ns'])
        dram_inst = MemoryInstance(name="dram", size=10000000000, r_bw=64, w_bw=64, r_cost=1, w_cost=1, area=0,
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

