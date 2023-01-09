import networkx as nx

from classes.stages.Stage import Stage
from classes.workload.dnn_workload import DNNWorkload
import classes.io.input_config as inputs
from classes.workload.dummy_layer_node import DummyNode
from utils import pickle_deepcopy
from networkx import DiGraph

import pandas as pd
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
        
        # Memory table
        self.mem_table = self.get_mem_from_table('inputs/cacti_memory.csv')

    def run(self):
        #TODO
        for i1 in range(self.mem_table.shape[0]-1):
            for i2 in range(i1, self.mem_table.shape[0]-1):
                for i3 in range(i2, self.mem_table.shape[0]-1):
                    # MODIFICATION OF THE ACCELERATOR OBJECT
                    updated_accelerator = self.update_accelerator_memory(self.accelerator,[self.mem_table['instance'][i1],self.mem_table['instance'][i2],self.mem_table['instance'][i3],self.mem_table['instance'][self.mem_table.shape[0]-1]])
                    # QUERY THE UNDERLYING STAGES WITH THE UPDATED ACCELERATOR
                    # We copy the kwargs just to be sure we don't accidentally keep unwanted changes
                    kwargs = pickle_deepcopy(self.kwargs)
                    kwargs["accelerator"] = updated_accelerator
                    sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
                    for cme, extra_info in sub_stage.run():
                        energy_total = cme.energy_total
                        logger.info(f"Total network energy for configuration # {i1+i2+i3} = {energy_total:.3e}")
                        yield cme, extra_info
                        # sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        
    @staticmethod
    def get_mem_from_table(table_file_path):
        mem_table = pd.read_csv(table_file_path)      # read mem config csv 
        for cfg_idx in range(mem_table.shape[0]):       # iterate over and build memory
            mem_table.loc[cfg_idx, 'instance'] = MemoryInstance(
                name=(mem_table['type'][cfg_idx] + '_' + str(mem_table['capacity'][cfg_idx]) + '_' + mem_table['node'][cfg_idx]),
                size=mem_table['capacity'][cfg_idx],
                r_bw=mem_table['r_bw'][cfg_idx],
                w_bw=mem_table['w_bw'][cfg_idx],
                r_cost=mem_table['r_cost'][cfg_idx],
                w_cost=mem_table['w_cost'][cfg_idx],
                r_port=mem_table['r_port'][cfg_idx],
                w_port=mem_table['w_port'][cfg_idx],
                rw_port=mem_table['rw_port'][cfg_idx],
                area=mem_table['area'][cfg_idx],
                latency=mem_table['latency'][cfg_idx],
            )
        return mem_table


    @staticmethod
    def update_accelerator_memory(accelerator, mem_config):
        accelerator_copy = pickle_deepcopy(accelerator)
        # As our accelerator can have different cores, iterate through all its cores
        for core in accelerator_copy.cores:
            # Construct new memory hierarchy
            memory_hierarchy_graph = MemoryHierarchy(core.memory_hierarchy.operational_array)
            memory_hierarchy_graph.add_memory(memory_instance=mem_config[0], operands=('I2',), served_dimensions={(0, 0)})
            memory_hierarchy_graph.add_memory(memory_instance=mem_config[1], operands=('O',), served_dimensions={(0, 1)})
            memory_hierarchy_graph.add_memory(memory_instance=mem_config[2], operands=('I1',), served_dimensions='all')
            memory_hierarchy_graph.add_memory(memory_instance=mem_config[3], operands=('I1', 'I2', 'O'), served_dimensions='all')

            core.memory_hierarchy = memory_hierarchy_graph

        return accelerator_copy

