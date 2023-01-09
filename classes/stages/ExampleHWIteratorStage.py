import networkx as nx

from classes.stages.Stage import Stage
from classes.workload.dnn_workload import DNNWorkload
import classes.io.input_config as inputs
from classes.workload.dummy_layer_node import DummyNode
from utils import pickle_deepcopy
from networkx import DiGraph

import logging
logger = logging.getLogger(__name__)


class ExampleHWIteratorStage(Stage):
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
        self.rf_energy_scalings = [1, 10, 100]

    def run(self):
        for rf_energy_scaling in self.rf_energy_scalings:
            # MODIFICATION OF THE ACCELERATOR OBJECT
            updated_accelerator = self.update_accelerator_energy_cost(self.accelerator, rf_energy_scaling)
            # QUERY THE UNDERLYING STAGES WITH THE UPDATED ACCELERATOR
            # We copy the kwargs just to be sure we don't accidentally keep unwanted changes
            kwargs = pickle_deepcopy(self.kwargs)
            kwargs["accelerator"] = updated_accelerator
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
            for cme, extra_info in sub_stage.run():
                energy_total = cme.energy_total
                logger.info(f"Total network energy for energy scaling {rf_energy_scaling} = {energy_total:.3e}")
                yield cme, extra_info
                # sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
    
    @staticmethod
    def update_accelerator_energy_cost(accelerator, rf_energy_scaling):
        accelerator_copy = pickle_deepcopy(accelerator)
        # As our accelerator can have different cores, iterate through all its cores
        for core in accelerator_copy.cores:
            # Find all memories inside the core's memory hierarchy
            # Here I do this based on the memory instances name
            memory_hierarchy = core.memory_hierarchy
            # The memory hierarchy is actually a NetworkX DiGraph consisting of "MemoryLevel" objects
            print(isinstance(memory_hierarchy, DiGraph))
            for memory_level in memory_hierarchy.nodes():
                if 'rf' in memory_level.name:  # in this example I just check the name
                    memory_level.read_energy *= rf_energy_scaling  # This should be done with a setter function
                    memory_level.write_energy *= rf_energy_scaling  # This should be done with a setter function
                    logger.info(f"Updated memory level {memory_level.name} energy.")
        return accelerator_copy