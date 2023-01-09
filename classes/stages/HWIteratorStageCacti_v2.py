# Libraries
import networkx as nx

from classes.stages.Stage import Stage
from classes.workload.dnn_workload import DNNWorkload
import classes.io.input_config as inputs
from classes.workload.dummy_layer_node import DummyNode
from utils import pickle_deepcopy
from networkx import DiGraph

import os
from classes.hardware.architecture.memory_instance import MemoryInstance
from classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from helper_scripts_rhyang.cacti import get_cacti

import logging
logger = logging.getLogger(__name__)


class HWIteratorStageCacti_v2(Stage):
    """
    Example stage that scales the energy cost of reading and writing a RF.
    For each HW design point we query the underlying stages and get back the optimal latency solution.
    In this example, I set the class (MemoryLevel) attributes directly. In practice, this should be done with a setter function.
    """
    def __init__(self, list_of_callables, *, accelerator, iterator_arch, node, **kwargs):
        """
        Initialization of self.workload.
        :param main_inputs: MainInputs, NOT copied
        """
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.iterator_arch = iterator_arch
        self.node = node


    def update_memory(self, accelerator, mem_config, node):
        accelerator_copy = pickle_deepcopy(accelerator)
        if(self.iterator_arch == "TPU_like"):
            assert(len(mem_config) == 6)
            rf_w_data = get_cacti(node, mem_config[0], mem_config[1], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            rf_o_data = get_cacti(node, mem_config[2], mem_config[3], 2, '/rsgs/scratch0/rhyang/codesign/cacti')
            sram_data = get_cacti(node, mem_config[4], mem_config[5], 1, '/rsgs/scratch0/rhyang/codesign/cacti')

            rf_w_inst = MemoryInstance(name="rf_w", size=(rf_w_data['size_bytes'])*8, r_bw=rf_w_data['bw'], w_bw=rf_w_data['bw'], r_cost=rf_w_data['cost'], w_cost=rf_w_data['cost'], area=0,
                                        r_port=1, w_port=1, rw_port=0, latency=rf_w_data['latency_ns'])
            rf_o_inst = MemoryInstance(name="rf_o", size=(rf_o_data['size_bytes'])*8, r_bw=2*(rf_o_data['bw']), w_bw=2*(rf_o_data['bw']), r_cost=rf_o_data['cost'], w_cost=rf_o_data['cost'], area=0,
                                        r_port=2, w_port=2, rw_port=0, latency=rf_o_data['latency_ns'])
            sram_inst = \
            MemoryInstance(name="sram", size=(sram_data['size_bytes'])*8, r_bw=sram_data['bw'], w_bw=sram_data['bw'], r_cost=sram_data['cost'], w_cost=sram_data['cost'], area=0,
                        r_port=1, w_port=1, rw_port=0, latency=sram_data['latency_ns'])
            dram_inst = MemoryInstance(name="dram", size=10000000000, r_bw=64, w_bw=64, r_cost=1, w_cost=1, area=0,
                                r_port=0, w_port=0, rw_port=1, latency=1)
            for core in accelerator_copy.cores:
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
        elif(self.iterator_arch == "Ascend_like"):
            assert(len(mem_config) == 14)
            reg_W1_data = get_cacti(node, mem_config[0], mem_config[1], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            reg_O1_data = get_cacti(node, mem_config[2], mem_config[3], 2, '/rsgs/scratch0/rhyang/codesign/cacti')
            sram_64KB_with_8_8K_64_1r_1w_I_data = get_cacti(node, mem_config[4], mem_config[5], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            sram_64KB_with_8_8K_256_1r_1w_W_data = get_cacti(node, mem_config[6], mem_config[7], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            sram_256KB_with_8_32KB_256_1r_1w_O_data = get_cacti(node, mem_config[8], mem_config[9], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            sram_1M_with_8_128K_bank_128_1r_1w_A_data = get_cacti(node, mem_config[10], mem_config[11], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            sram_1M_with_8_128K_bank_128_1r_1w_W_data = get_cacti(node, mem_config[12], mem_config[13], 1, '/rsgs/scratch0/rhyang/codesign/cacti')

            reg_W1 = MemoryInstance(name="rf_1B", size=(reg_W1_data['size_bytes'])*8, r_bw=reg_W1_data['bw'], w_bw=reg_W1_data['bw'], r_cost=reg_W1_data['cost'], w_cost=reg_W1_data['cost'], area=0,
                                    r_port=1, w_port=1, rw_port=0, latency=reg_W1_data['latency_ns'])
            reg_O1 = MemoryInstance(name="rf_2B", size=(reg_O1_data['size_bytes'])*8, r_bw=reg_O1_data['bw'], w_bw=reg_O1_data['bw'], r_cost=reg_O1_data['cost'], w_cost=reg_O1_data['cost'], area=0,
                                    r_port=2, w_port=2, rw_port=0, latency=reg_O1_data['latency_ns'])
            sram_64KB_with_8_8K_64_1r_1w_I = \
                MemoryInstance(name="sram_64KB_I", size=(sram_64KB_with_8_8K_64_1r_1w_I_data['size_bytes'])*8, r_bw=sram_64KB_with_8_8K_64_1r_1w_I_data['bw'], w_bw=sram_64KB_with_8_8K_64_1r_1w_I_data['bw'], r_cost=sram_64KB_with_8_8K_64_1r_1w_I_data['cost'], w_cost=sram_64KB_with_8_8K_64_1r_1w_I_data['cost'], area=0,
                            r_port=1, w_port=1, rw_port=0, latency=sram_64KB_with_8_8K_64_1r_1w_I_data['latency_ns'], min_r_granularity=64, min_w_granularity=64)
            sram_64KB_with_8_8K_256_1r_1w_W = \
                MemoryInstance(name="sram_64KB_W", size=(sram_64KB_with_8_8K_256_1r_1w_W_data['size_bytes'])*8, r_bw=sram_64KB_with_8_8K_256_1r_1w_W_data['bw'], w_bw=sram_64KB_with_8_8K_256_1r_1w_W_data['bw'], r_cost=sram_64KB_with_8_8K_256_1r_1w_W_data['cost'], w_cost=sram_64KB_with_8_8K_256_1r_1w_W_data['cost'], area=0,
                            r_port=1, w_port=1, rw_port=0, latency=sram_64KB_with_8_8K_256_1r_1w_W_data['latency_ns'], min_r_granularity=64, min_w_granularity=64)
            sram_256KB_with_8_32KB_256_1r_1w_O = \
                MemoryInstance(name="sram_256KB_O", size=(sram_256KB_with_8_32KB_256_1r_1w_O_data['size_bytes'])*8, r_bw=sram_256KB_with_8_32KB_256_1r_1w_O_data['bw'], w_bw=sram_256KB_with_8_32KB_256_1r_1w_O_data['bw'], r_cost=sram_256KB_with_8_32KB_256_1r_1w_O_data['cost'], w_cost=sram_256KB_with_8_32KB_256_1r_1w_O_data['cost'], area=0,
                            r_port=1, w_port=1, rw_port=0, latency=sram_256KB_with_8_32KB_256_1r_1w_O_data['latency_ns'], min_r_granularity=64, min_w_granularity=64)
            sram_1M_with_8_128K_bank_128_1r_1w_A = \
                MemoryInstance(name="sram_1MB_A", size=(sram_1M_with_8_128K_bank_128_1r_1w_A_data['size_bytes'])*8, r_bw=sram_1M_with_8_128K_bank_128_1r_1w_A_data['bw'], w_bw=sram_1M_with_8_128K_bank_128_1r_1w_A_data['bw'], r_cost=sram_1M_with_8_128K_bank_128_1r_1w_A_data['cost'], w_cost=sram_1M_with_8_128K_bank_128_1r_1w_A_data['cost'], area=0,
                            r_port=1, w_port=1, rw_port=0, latency=sram_1M_with_8_128K_bank_128_1r_1w_A_data['latency_ns'], min_r_granularity=64, min_w_granularity=64)
            sram_1M_with_8_128K_bank_128_1r_1w_W = \
                MemoryInstance(name="sram_1MB_W", size=(sram_1M_with_8_128K_bank_128_1r_1w_W_data['size_bytes'])*8, r_bw=sram_1M_with_8_128K_bank_128_1r_1w_W_data['bw'], w_bw=sram_1M_with_8_128K_bank_128_1r_1w_W_data['bw'], r_cost=sram_1M_with_8_128K_bank_128_1r_1w_W_data['cost'], w_cost=sram_1M_with_8_128K_bank_128_1r_1w_W_data['cost'], area=0,
                            r_port=1, w_port=1, rw_port=0, latency=sram_1M_with_8_128K_bank_128_1r_1w_W_data['latency_ns'], min_r_granularity=64, min_w_granularity=64)
            dram = MemoryInstance(name="dram", size=10000000000, r_bw=64, w_bw=64, r_cost=1, w_cost=1, area=0,
                                r_port=0, w_port=0, rw_port=1, latency=1)
            for core in accelerator_copy.cores:
                memory_hierarchy_graph = MemoryHierarchy(core.memory_hierarchy.operational_array)
                memory_hierarchy_graph.add_memory(memory_instance=reg_W1_data, operands=('I2',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                                served_dimensions={(0, 0, 1, 0), (0, 0, 0, 1)})
                memory_hierarchy_graph.add_memory(memory_instance=reg_O1_data, operands=('O',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_2', 'th': 'r_port_2'},),
                                                served_dimensions={(0, 1, 0, 0)})
                memory_hierarchy_graph.add_memory(memory_instance=sram_64KB_with_8_8K_256_1r_1w_W_data, operands=('I2',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                                served_dimensions='all')
                memory_hierarchy_graph.add_memory(memory_instance=sram_64KB_with_8_8K_64_1r_1w_I_data, operands=('I1',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                                served_dimensions='all')
                memory_hierarchy_graph.add_memory(memory_instance=sram_256KB_with_8_32KB_256_1r_1w_O_data, operands=('O',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_1', 'th': 'r_port_1'},),
                                                served_dimensions='all')
                memory_hierarchy_graph.add_memory(memory_instance=sram_1M_with_8_128K_bank_128_1r_1w_W_data, operands=('I2',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                                served_dimensions='all')
                memory_hierarchy_graph.add_memory(memory_instance=sram_1M_with_8_128K_bank_128_1r_1w_A_data, operands=('I1', 'O'),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},
                                                            {'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_1', 'th': 'r_port_1'},),
                                                served_dimensions='all')
                memory_hierarchy_graph.add_memory(memory_instance=dram, operands=('I1', 'I2', 'O'),
                                                port_alloc=({'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},
                                                            {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},
                                                            {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': 'rw_port_1', 'th': 'rw_port_1'},),
                                                served_dimensions='all')
                core.memory_hierarchy = memory_hierarchy_graph
                core.check_valid()
                core.recalculate_memory_hierarchy_information()
        elif(self.iterator_arch == "Edge_TPU_like"):
            assert(len(mem_config) == 8)
            reg_IW1_data = get_cacti(node, mem_config[0], mem_config[1], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            reg_O1_data = get_cacti(node, mem_config[2], mem_config[3], 2, '/rsgs/scratch0/rhyang/codesign/cacti')
            sram_32KB_512_1r_1w_data = get_cacti(node, mem_config[4], mem_config[5], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            sram_2M_with_16_128K_bank_128_1r_1w_data = get_cacti(node, mem_config[6], mem_config[7], 1, '/rsgs/scratch0/rhyang/codesign/cacti')

            reg_IW1 = MemoryInstance(name="rf_1B", size=(reg_IW1_data['size_bytes'])*8, r_bw=reg_IW1_data['bw'], w_bw=reg_IW1_data['bw'], r_cost=reg_IW1_data['cost'], w_cost=reg_IW1_data['cost'], area=0,
                                    r_port=1, w_port=1, rw_port=0, latency=reg_IW1_data['latency_ns'])
            reg_O1 = MemoryInstance(name="rf_2B", size=(reg_O1_data['size_bytes'])*8, r_bw=reg_O1_data['bw'], w_bw=reg_O1_data['bw'], r_cost=reg_O1_data['cost'], w_cost=reg_O1_data['cost'], area=0,
                                    r_port=2, w_port=2, rw_port=0, latency=reg_O1_data['latency_ns'])
            sram_32KB_512_1r_1w = \
                MemoryInstance(name="sram_32KB", size=(sram_32KB_512_1r_1w_data['size_bytes'])*8, r_bw=sram_32KB_512_1r_1w_data['bw'], w_bw=sram_32KB_512_1r_1w_data['bw'], r_cost=sram_32KB_512_1r_1w_data['cost'], w_cost=sram_32KB_512_1r_1w_data['cost'], area=0,
                            r_port=1, w_port=1, rw_port=0, latency=sram_32KB_512_1r_1w_data['latency_ns'], min_r_granularity=64, min_w_granularity=64)
            sram_2M_with_16_128K_bank_128_1r_1w = \
                MemoryInstance(name="sram_2MB", size=(sram_2M_with_16_128K_bank_128_1r_1w_data['size_bytes'])*8, r_bw=sram_2M_with_16_128K_bank_128_1r_1w_data['bw'], w_bw=sram_2M_with_16_128K_bank_128_1r_1w_data['bw'], r_cost=sram_2M_with_16_128K_bank_128_1r_1w_data['cost'], w_cost=sram_2M_with_16_128K_bank_128_1r_1w_data['cost'], area=0,
                            r_port=1, w_port=1, rw_port=0, latency=sram_2M_with_16_128K_bank_128_1r_1w_data['latency_ns'], min_r_granularity=64, min_w_granularity=64)
            dram = MemoryInstance(name="dram", size=10000000000, r_bw=64, w_bw=64, r_cost=1, w_cost=1, area=0,
                                r_port=0, w_port=0, rw_port=1, latency=1)
            for core in accelerator_copy.cores:
                memory_hierarchy_graph = MemoryHierarchy(core.memory_hierarchy.operational_array)
                memory_hierarchy_graph.add_memory(memory_instance=reg_IW1, operands=('I2',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                                served_dimensions={(0, 0, 1, 0), (0, 0, 0, 1)})
                memory_hierarchy_graph.add_memory(memory_instance=reg_O1, operands=('O',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_2', 'th': 'r_port_2'},),
                                                served_dimensions={(0, 1, 0, 0)})
                memory_hierarchy_graph.add_memory(memory_instance=sram_32KB_512_1r_1w, operands=('I2',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                                served_dimensions='all')
                memory_hierarchy_graph.add_memory(memory_instance=sram_2M_with_16_128K_bank_128_1r_1w, operands=('I1', 'O'),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},
                                                            {'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_1', 'th': 'r_port_1'},),
                                                served_dimensions='all')
                memory_hierarchy_graph.add_memory(memory_instance=dram, operands=('I1', 'I2', 'O'),
                                                port_alloc=({'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},
                                                            {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},
                                                            {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': 'rw_port_1', 'th': 'rw_port_1'},),
                                                served_dimensions='all')
                core.memory_hierarchy = memory_hierarchy_graph
                core.check_valid()
                core.recalculate_memory_hierarchy_information()
        elif(self.iterator_arch == "Eyeriss_like"):
            assert(len(mem_config) == 10)
            rf1_data = get_cacti(node, mem_config[0], mem_config[1], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            rf2_data = get_cacti(node, mem_config[2], mem_config[3], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            lb2_data = get_cacti(node, mem_config[4], mem_config[5], 2, '/rsgs/scratch0/rhyang/codesign/cacti')
            lb2_64KB_data = get_cacti(node, mem_config[6], mem_config[7], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            gb_data = get_cacti(node, mem_config[8], mem_config[9], 2, '/rsgs/scratch0/rhyang/codesign/cacti')
            
            rf1 = MemoryInstance(name="rf_64B", size=(rf1_data['size_bytes'])*8, r_bw=rf1_data['bw'], w_bw=rf1_data['bw'], r_cost=rf1_data['cost'], w_cost=rf1_data['cost'], area=0.3, r_port=1, w_port=1, rw_port=0, latency=rf1_data['latency_ns'])  # rd E per bit 0.125
            rf2 = MemoryInstance(name="rf_16B", size=(rf2_data['size_bytes'])*8, r_bw=rf2_data['bw'], w_bw=rf2_data['bw'], r_cost=rf2_data['cost'], w_cost=rf2_data['cost'], area=0.95, r_port=1, w_port=1, rw_port=1, latency=rf2_data['latency_ns'])  # rd E per bit 0.0625
            lb2 = MemoryInstance(name="sram_8KB", size=(lb2_data['size_bytes'])*8, r_bw=lb2_data['bw'], w_bw=lb2_data['bw'], r_cost=lb2_data['cost'], w_cost=lb2_data['cost'], r_port=0, area=3, w_port=0, rw_port=2, latency=lb2_data['latency_ns'])  # rd E per bit 0.08
            lb2_64KB = MemoryInstance(name="sram_64KB", size=(lb2_64KB_data['size_bytes'])*8, r_bw=lb2_64KB_data['bw'], w_bw=lb2_64KB_data['bw'], r_cost=lb2_64KB_data['cost'], w_cost=lb2_64KB_data['cost'], area=6, r_port=1, w_port=1, rw_port=0, latency=lb2_64KB_data['latency_ns'])  # rd E per bit 0.08
            gb = MemoryInstance(name="sram_1M", size=(gb_data['size_bytes'])*8, r_bw=gb_data['bw'], w_bw=gb_data['bw'], r_cost=gb_data['cost'], w_cost=['cost'], area=25, r_port=0, w_port=0, rw_port=2, latency=['latency_ns'])  # rd E per bit 0.26
            dram = MemoryInstance(name="dram", size=10000000000, r_bw=64, w_bw=64, r_cost=1, w_cost=1, area=0, r_port=0, w_port=0, rw_port=1, latency=1) # rd E per bit 16
            for core in accelerator_copy.cores:
                memory_hierarchy_graph = MemoryHierarchy(core.memory_hierarchy.operational_array)
                memory_hierarchy_graph.add_memory(memory_instance=rf1, operands=('I1',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                                served_dimensions=set())
                memory_hierarchy_graph.add_memory(memory_instance=rf1, operands=('I2',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                                served_dimensions=set())
                memory_hierarchy_graph.add_memory(memory_instance=rf2, operands=('O',),
                                                port_alloc=({'fh': 'rw_port_1', 'tl': 'r_port_1', 'fl': 'w_port_1', 'th': 'rw_port_1'},),
                                                served_dimensions=set())
                memory_hierarchy_graph.add_memory(memory_instance=lb2, operands=('O',),
                                                port_alloc=({'fh': 'rw_port_1', 'tl': 'rw_port_2', 'fl': 'rw_port_2', 'th': 'rw_port_1'},),
                                                served_dimensions='all', )
                memory_hierarchy_graph.add_memory(memory_instance=lb2_64KB, operands=('I2',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                                served_dimensions='all')
                memory_hierarchy_graph.add_memory(memory_instance=gb, operands=('I1', 'O'),
                                                port_alloc=({'fh': 'rw_port_1', 'tl': 'rw_port_2', 'fl': None, 'th': None},
                                                            {'fh': 'rw_port_1', 'tl': 'rw_port_2', 'fl': 'rw_port_2', 'th': 'rw_port_1'},),
                                                served_dimensions='all')
                memory_hierarchy_graph.add_memory(memory_instance=dram, operands=('I1', 'I2', 'O'),
                                                port_alloc=({'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},
                                                            {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},
                                                            {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': 'rw_port_1', 'th': 'rw_port_1'},),
                                                served_dimensions='all')
                core.memory_hierarchy = memory_hierarchy_graph
                core.check_valid()
                core.recalculate_memory_hierarchy_information()
        elif(self.iterator_arch == "Meta_prototype"):
            assert(len(mem_config) == 12)
            reg_IW1_data = get_cacti(node, mem_config[0], mem_config[1], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            reg_O1_data = get_cacti(node, mem_config[2], mem_config[3], 2, '/rsgs/scratch0/rhyang/codesign/cacti')
            sram_64KB_with_8_8K_64_1r_1w_data = get_cacti(node, mem_config[4], mem_config[5], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            sram_32KB_with_4_8K_64_1r_1w_data = get_cacti(node, mem_config[6], mem_config[7], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            sram_1M_with_8_128K_bank_128_1r_1w_A_data = get_cacti(node, mem_config[8], mem_config[9], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            sram_1M_with_8_128K_bank_128_1r_1w_W_data = get_cacti(node, mem_config[10], mem_config[11], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            
            reg_IW1 = MemoryInstance(name="rf_1B", size=(reg_IW1_data['size_bytes'])*8, r_bw=reg_IW1_data['bw'], w_bw=reg_IW1_data['bw'], r_cost=reg_IW1_data['cost'], w_cost=reg_IW1_data['cost'], area=0,
                                    r_port=1, w_port=1, rw_port=0, latency=reg_IW1_data['latency_ns'])
            reg_O1 = MemoryInstance(name="rf_2B", size=(reg_O1_data['size_bytes'])*8, r_bw=reg_O1_data['bw'], w_bw=reg_O1_data['bw'], r_cost=reg_O1_data['cost'], w_cost=reg_O1_data['cost'], area=0,
                                    r_port=2, w_port=2, rw_port=0, latency=reg_O1_data['latency_ns'])
            sram_64KB_with_8_8K_64_1r_1w = \
                MemoryInstance(name="sram_64KB", size=(sram_64KB_with_8_8K_64_1r_1w_data['size_bytes'])*8, r_bw=sram_64KB_with_8_8K_64_1r_1w_data['bw'], w_bw=sram_64KB_with_8_8K_64_1r_1w_data['bw'], r_cost=sram_64KB_with_8_8K_64_1r_1w_data['cost'], w_cost=sram_64KB_with_8_8K_64_1r_1w_data['cost'], area=0,
                            r_port=1, w_port=1, rw_port=0, latency=sram_64KB_with_8_8K_64_1r_1w_data['latency_ns'], min_r_granularity=64, min_w_granularity=64)
            sram_32KB_with_4_8K_64_1r_1w = \
                MemoryInstance(name="sram_32KB", size=(sram_32KB_with_4_8K_64_1r_1w_data['size_bytes'])*8, r_bw=sram_32KB_with_4_8K_64_1r_1w_data['bw'], w_bw=sram_32KB_with_4_8K_64_1r_1w_data['bw'], r_cost=sram_32KB_with_4_8K_64_1r_1w_data['cost'], w_cost=sram_32KB_with_4_8K_64_1r_1w_data['cost'], area=0,
                            r_port=1, w_port=1, rw_port=0, latency=sram_32KB_with_4_8K_64_1r_1w_data['latency_ns'], min_r_granularity=64, min_w_granularity=64)
            sram_1M_with_8_128K_bank_128_1r_1w_A = \
                MemoryInstance(name="sram_1MB_A", size=(sram_1M_with_8_128K_bank_128_1r_1w_A_data['size_bytes'])*8, r_bw=sram_1M_with_8_128K_bank_128_1r_1w_A_data['bw'], w_bw=sram_1M_with_8_128K_bank_128_1r_1w_A_data['bw'], r_cost=sram_1M_with_8_128K_bank_128_1r_1w_A_data['cost'], w_cost=sram_1M_with_8_128K_bank_128_1r_1w_A_data['cost'], area=0,
                            r_port=1, w_port=1, rw_port=0, latency=sram_1M_with_8_128K_bank_128_1r_1w_A_data['latency_ns'], min_r_granularity=64, min_w_granularity=64)
            sram_1M_with_8_128K_bank_128_1r_1w_W = \
                MemoryInstance(name="sram_1MB_W", size=(sram_1M_with_8_128K_bank_128_1r_1w_W_data['size_bytes'])*8, r_bw=sram_1M_with_8_128K_bank_128_1r_1w_W_data['bw'], w_bw=sram_1M_with_8_128K_bank_128_1r_1w_W_data['bw'], r_cost=sram_1M_with_8_128K_bank_128_1r_1w_W_data['cost'], w_cost=sram_1M_with_8_128K_bank_128_1r_1w_W_data['cost'], area=0,
                            r_port=1, w_port=1, rw_port=0, latency=sram_1M_with_8_128K_bank_128_1r_1w_W_data['latency_ns'], min_r_granularity=64, min_w_granularity=64)
            dram = MemoryInstance(name="dram", size=10000000000, r_bw=64, w_bw=64, r_cost=1, w_cost=1, area=0,
                                r_port=0, w_port=0, rw_port=1, latency=1)
            for core in accelerator_copy.cores:
                memory_hierarchy_graph = MemoryHierarchy(core.memory_hierarchy.operational_array)
                memory_hierarchy_graph.add_memory(memory_instance=reg_IW1, operands=('I2',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                                served_dimensions={(0, 0, 1, 0), (0, 0, 0, 1)})
                memory_hierarchy_graph.add_memory(memory_instance=reg_O1, operands=('O',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_2', 'th': 'r_port_2'},),
                                                served_dimensions={(0, 1, 0, 0)})
                memory_hierarchy_graph.add_memory(memory_instance=sram_64KB_with_8_8K_64_1r_1w, operands=('I2',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                                served_dimensions='all')
                memory_hierarchy_graph.add_memory(memory_instance=sram_1M_with_8_128K_bank_128_1r_1w_W, operands=('I2',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                                served_dimensions='all')
                memory_hierarchy_graph.add_memory(memory_instance=sram_32KB_with_4_8K_64_1r_1w, operands=('I1',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                                served_dimensions='all')
                memory_hierarchy_graph.add_memory(memory_instance=sram_1M_with_8_128K_bank_128_1r_1w_A, operands=('I1', 'O'),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},
                                                            {'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_1', 'th': 'r_port_1'},),
                                                served_dimensions='all')
                memory_hierarchy_graph.add_memory(memory_instance=dram, operands=('I1', 'I2', 'O'),
                                                port_alloc=({'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},
                                                            {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},
                                                            {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': 'rw_port_1', 'th': 'rw_port_1'},),
                                                served_dimensions='all')
                core.memory_hierarchy = memory_hierarchy_graph
                core.check_valid()
                core.recalculate_memory_hierarchy_information()
        elif(self.iterator_arch == "Tesla_NPU_like"):
            assert(len(mem_config) == 12)
            reg_W1_data = get_cacti(node, mem_config[0], mem_config[1], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            reg_O4_data = get_cacti(node, mem_config[2], mem_config[3], 2, '/rsgs/scratch0/rhyang/codesign/cacti')
            sram_1KB_256_1r_1w_I_data = get_cacti(node, mem_config[4], mem_config[5], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            sram_1KB_256_1r_1w_W_data = get_cacti(node, mem_config[6], mem_config[7], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            sram_1M_with_8_128K_bank_128_1r_1w_A_data = get_cacti(node, mem_config[8], mem_config[9], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            sram_1M_with_8_128K_bank_128_1r_1w_W_data = get_cacti(node, mem_config[10], mem_config[11], 1, '/rsgs/scratch0/rhyang/codesign/cacti')
            
            reg_W1 = MemoryInstance(name="rf_1B", size=(reg_W1_data['size_bytes'])*8, r_bw=reg_W1_data['bw'], w_bw=reg_W1_data['bw'], r_cost=reg_W1_data['cost'], w_cost=reg_W1_data['cost'], area=0,
                                    r_port=1, w_port=1, rw_port=0, latency=reg_W1_data['latency_ns'])
            reg_O4 = MemoryInstance(name="rf_4B", size=(reg_O4_data['size_bytes'])*8, r_bw=reg_O4_data['bw'], w_bw=reg_O4_data['bw'], r_cost=reg_O4_data['cost'], w_cost=reg_O4_data['cost'], area=0,
                                    r_port=2, w_port=2, rw_port=0, latency=reg_O4_data['latency_ns'])
            sram_1KB_256_1r_1w_I = \
                MemoryInstance(name="sram_1KB_I", size=(sram_1KB_256_1r_1w_I_data['size_bytes'])*8, r_bw=sram_1KB_256_1r_1w_I_data['bw'], w_bw=sram_1KB_256_1r_1w_I_data['bw'], r_cost=sram_1KB_256_1r_1w_I_data['cost'], w_cost=sram_1KB_256_1r_1w_I_data['cost'], area=0,
                            r_port=1, w_port=1, rw_port=0, latency=sram_1KB_256_1r_1w_I_data['latency_ns'], min_r_granularity=64, min_w_granularity=64)
            sram_1KB_256_1r_1w_W = \
                MemoryInstance(name="sram_1KB_W", size=(sram_1KB_256_1r_1w_W_data['size_bytes'])*8, r_bw=sram_1KB_256_1r_1w_W_data['bw'], w_bw=sram_1KB_256_1r_1w_W_data['bw'], r_cost=sram_1KB_256_1r_1w_W_data['cost'], w_cost=sram_1KB_256_1r_1w_W_data['cost'], area=0,
                            r_port=1, w_port=1, rw_port=0, latency=sram_1KB_256_1r_1w_W_data['latency_ns'], min_r_granularity=64, min_w_granularity=64)
            sram_1M_with_8_128K_bank_128_1r_1w_A = \
                MemoryInstance(name="sram_1MB_A", size=(sram_1M_with_8_128K_bank_128_1r_1w_A_data['size_bytes'])*8, r_bw=sram_1M_with_8_128K_bank_128_1r_1w_A_data['bw'], w_bw=sram_1M_with_8_128K_bank_128_1r_1w_A_data['bw'], r_cost=sram_1M_with_8_128K_bank_128_1r_1w_A_data['cost'], w_cost=sram_1M_with_8_128K_bank_128_1r_1w_A_data['cost'], area=0,
                            r_port=1, w_port=1, rw_port=0, latency=sram_1M_with_8_128K_bank_128_1r_1w_A_data['latency_ns'], min_r_granularity=64, min_w_granularity=64)
            sram_1M_with_8_128K_bank_128_1r_1w_W = \
                MemoryInstance(name="sram_1MB_W", size=(sram_1M_with_8_128K_bank_128_1r_1w_W_data['size_bytes'])*8, r_bw=sram_1M_with_8_128K_bank_128_1r_1w_W_data['bw'], w_bw=sram_1M_with_8_128K_bank_128_1r_1w_W_data['bw'], r_cost=sram_1M_with_8_128K_bank_128_1r_1w_W_data['cost'], w_cost=sram_1M_with_8_128K_bank_128_1r_1w_W_data['cost'], area=0,
                            r_port=1, w_port=1, rw_port=0, latency=sram_1M_with_8_128K_bank_128_1r_1w_W_data['latency_ns'], min_r_granularity=64, min_w_granularity=64)
            dram = MemoryInstance(name="dram", size=10000000000, r_bw=64, w_bw=64, r_cost=700, w_cost=750, area=0,
                                r_port=0, w_port=0, rw_port=1, latency=1)
            for core in accelerator_copy.cores:
                memory_hierarchy_graph = MemoryHierarchy(core.memory_hierarchy.operational_array)
                memory_hierarchy_graph.add_memory(memory_instance=reg_W1, operands=('I2',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                                served_dimensions={(0, 1, 0), (0, 0, 1)})
                memory_hierarchy_graph.add_memory(memory_instance=reg_O4, operands=('O',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_2', 'th': 'r_port_2'},),
                                                served_dimensions={(0, 0, 0)})
                memory_hierarchy_graph.add_memory(memory_instance=sram_1KB_256_1r_1w_I, operands=('I1',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                                served_dimensions='all')
                memory_hierarchy_graph.add_memory(memory_instance=sram_1KB_256_1r_1w_W, operands=('I2',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                                served_dimensions='all')
                memory_hierarchy_graph.add_memory(memory_instance=sram_1M_with_8_128K_bank_128_1r_1w_W, operands=('I2',),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
                                                served_dimensions='all')
                memory_hierarchy_graph.add_memory(memory_instance=sram_1M_with_8_128K_bank_128_1r_1w_A, operands=('I1', 'O'),
                                                port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},
                                                            {'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_1', 'th': 'r_port_1'},),
                                                served_dimensions='all')
                memory_hierarchy_graph.add_memory(memory_instance=dram, operands=('I1', 'I2', 'O'),
                                                port_alloc=({'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},
                                                            {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},
                                                            {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': 'rw_port_1', 'th': 'rw_port_1'},),
                                                served_dimensions='all')
                core.memory_hierarchy = memory_hierarchy_graph
                core.check_valid()
                core.recalculate_memory_hierarchy_information()
        else:
            print("Iterator Arch incorrect")  # error

        return accelerator_copy        


    def run(self):
        if(self.iterator_arch == "TPU_like"):
            for rf_w_size in [128,256,512,1024]:
                for rf_w_bw in [8,16,32,64]:
                    for rf_o_size in [128,256,512,1024]:
                        for rf_o_bw in [8,16,32,64]:
                            for sram_size in [x*rf_w_size*rf_w_size for x in range(1,5)]:
                                for sram_bw in [32,64]:
                                    updated_accelerator = self.update_memory(self.accelerator, [rf_w_size, rf_w_bw, rf_o_size, rf_o_bw, sram_size, sram_bw], self.node)
                                    kwargs = pickle_deepcopy(self.kwargs)
                                    kwargs["accelerator"] = updated_accelerator
                                    sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
                                    for cme, extra_info in sub_stage.run():
                                        energy_total = cme.energy_total
                                        yield cme, extra_info
        
        elif(self.iterator_arch == "Ascend_like"):
            for reg_W1_size in [1,2,4]:
                for reg_W1_bw in [reg_W1_size*8]:
                    for reg_O1_size in [2*reg_W1_size, 4*reg_W1_size]:  # REG for O is double the REG for W
                        for reg_O1_bw in [reg_O1_size*8]:
                            for sram_64KB_with_8_8K_64_1r_1w_I_size in [x*reg_W1_size*1024 for x in [4,8,12,16]]:
                                for sram_64KB_with_8_8K_64_1r_1w_I_bw in [32,64,128]:
                                    for sram_64KB_with_8_8K_256_1r_1w_W_size in [sram_64KB_with_8_8K_64_1r_1w_I_size]:  # SRAM for W follows SRAM for I
                                        for sram_64KB_with_8_8K_256_1r_1w_W_bw in [x*sram_64KB_with_8_8K_64_1r_1w_I_bw for x in [1,2,4]]:
                                            for sram_256KB_with_8_32KB_256_1r_1w_O_size in [x*sram_64KB_with_8_8K_64_1r_1w_I_size for x in [8,16,32,64]]:
                                                for sram_256KB_with_8_32KB_256_1r_1w_O_bw in [x*sram_64KB_with_8_8K_256_1r_1w_W_bw for x in [0.5,1,2,4]]:
                                                    for sram_1M_with_8_128K_bank_128_1r_1w_A_size in [x*sram_256KB_with_8_32KB_256_1r_1w_O_size for x in [2,4,8]]:
                                                        for sram_1M_with_8_128K_bank_128_1r_1w_A_bw in [x*sram_256KB_with_8_32KB_256_1r_1w_O_bw for x in [1,2,4]]:
                                                            for sram_1M_with_8_128K_bank_128_1r_1w_W_size in [x*sram_1M_with_8_128K_bank_128_1r_1w_A_size for x in [0.5,1,2]]:
                                                                for sram_1M_with_8_128K_bank_128_1r_1w_W_bw in [x*sram_1M_with_8_128K_bank_128_1r_1w_A_bw for x in [0.5,1,2]]:     
                                                                    mem_config = [
                                                                        reg_W1_size, reg_W1_bw,
                                                                        reg_O1_size, reg_O1_bw,
                                                                        sram_64KB_with_8_8K_64_1r_1w_I_size, sram_64KB_with_8_8K_64_1r_1w_I_bw,
                                                                        sram_64KB_with_8_8K_256_1r_1w_W_size, sram_64KB_with_8_8K_256_1r_1w_W_bw,
                                                                        sram_256KB_with_8_32KB_256_1r_1w_O_size, sram_256KB_with_8_32KB_256_1r_1w_O_bw,
                                                                        sram_1M_with_8_128K_bank_128_1r_1w_A_size, sram_1M_with_8_128K_bank_128_1r_1w_A_bw,
                                                                        sram_1M_with_8_128K_bank_128_1r_1w_W_size, sram_1M_with_8_128K_bank_128_1r_1w_W_bw
                                                                    ]                    
                                                                    updated_accelerator = self.update_memory(self.accelerator, mem_config, self.node)
                                                                    kwargs = pickle_deepcopy(self.kwargs)
                                                                    kwargs["accelerator"] = updated_accelerator
                                                                    sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
                                                                    for cme, extra_info in sub_stage.run():
                                                                        energy_total = cme.energy_total
                                                                        yield cme, extra_info
                                        
        elif(self.iterator_arch == "Edge_TPU_like"):
            for reg_IW1_size in [1,2,4]:
                for reg_IW1_bw in [reg_IW1_size*8]:
                    for reg_O1_size in [2*reg_IW1_size, 4*reg_IW1_size]:
                        for reg_O1_bw in [reg_O1_size*8]:
                            for sram_32KB_512_1r_1w_size in [x*reg_IW1_size*1024 for x in [8,16,32,64,128]]:
                                for sram_32KB_512_1r_1w_bw in [64,128,256,512,1024]:
                                    for sram_2M_with_16_128K_bank_128_1r_1w_size in [x*sram_32KB_512_1r_1w_size for x in [2,4,8]]:
                                        for sram_2M_with_16_128K_bank_128_1r_1w_bw in [256,512,1024,2048]:
                                            mem_config = [
                                                reg_IW1_size, reg_IW1_bw,
                                                reg_O1_size, reg_O1_bw,
                                                sram_32KB_512_1r_1w_size, sram_32KB_512_1r_1w_bw,
                                                sram_2M_with_16_128K_bank_128_1r_1w_size, sram_2M_with_16_128K_bank_128_1r_1w_bw
                                            ]
                                            updated_accelerator = self.update_memory(self.accelerator, mem_config, self.node)
                                            kwargs = pickle_deepcopy(self.kwargs)
                                            kwargs["accelerator"] = updated_accelerator
                                            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
                                            for cme, extra_info in sub_stage.run():
                                                energy_total = cme.energy_total
                                                yield cme, extra_info
        elif(self.iterator_arch == "Eyeriss_like"):
            for rf1_size in [16,32,64,128]:
                for rf1_bw in [8,16]:
                    for rf2_size in [16,32,64,128]:
                        for rf2_bw in [8,16]:
                            for lb2_size in [x*1024 for x in [4,8,16,32]]:
                                for lb2_bw in [32,64,128,256]:
                                    for lb2_64KB_size in [x*1024 for x in [32,64,128]]:
                                        for lb2_64KB_bw in [32,64,128,256]:
                                            for gb_size in [x*1024*1024 for x in [0.5,1,2]]:
                                                for gb_bw in [x*384 for x in [0.5,1,2]]:
                                                    mem_config = [
                                                        rf1_size, rf1_bw,
                                                        rf2_size, rf2_bw,
                                                        lb2_size, lb2_bw,
                                                        lb2_64KB_size, lb2_64KB_bw,
                                                        gb_size, gb_bw
                                                    ]
                                                    updated_accelerator = self.update_memory(self.accelerator, mem_config, self.node)
                                                    kwargs = pickle_deepcopy(self.kwargs)
                                                    kwargs["accelerator"] = updated_accelerator
                                                    sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
                                                    for cme, extra_info in sub_stage.run():
                                                        energy_total = cme.energy_total
                                                        yield cme, extra_info
        elif(self.iterator_arch == "Meta_prototype"):
            for reg_IW1_size in [1,2,4]:
                for reg_IW1_bw in [8,16]:
                    for reg_O1_size in [reg_IW1_size*2]:
                        for reg_O1_bw in [reg_IW1_bw*2]:
                            for sram_64KB_with_8_8K_64_1r_1w_size in [x*1024 for x in [32,64,128]]:
                                for sram_64KB_with_8_8K_64_1r_1w_bw in [x*8 for x in [32,64,128]]:
                                    for sram_32KB_with_4_8K_64_1r_1w_size in [x*1024 for x in [16,32,64,128]]:
                                        for sram_32KB_with_4_8K_64_1r_1w_bw in [x*8 for x in [16,32,64,128]]:
                                            for sram_1M_with_8_128K_bank_128_1r_1w_A_size in [x*1024 for x in [64,128,256]]:
                                                for sram_1M_with_8_128K_bank_128_1r_1w_A_bw in [x*8 for x in [64,128,256]]:
                                                    for sram_1M_with_8_128K_bank_128_1r_1w_W_size in [sram_1M_with_8_128K_bank_128_1r_1w_A_size]:
                                                        for sram_1M_with_8_128K_bank_128_1r_1w_W_bw in [sram_1M_with_8_128K_bank_128_1r_1w_A_bw]:
                                                            mem_config = [
                                                                reg_IW1_size, reg_IW1_bw,
                                                                reg_O1_size, reg_O1_bw,
                                                                sram_64KB_with_8_8K_64_1r_1w_size, sram_64KB_with_8_8K_64_1r_1w_bw,
                                                                sram_32KB_with_4_8K_64_1r_1w_size, sram_32KB_with_4_8K_64_1r_1w_bw,
                                                                sram_1M_with_8_128K_bank_128_1r_1w_A_size, sram_1M_with_8_128K_bank_128_1r_1w_A_bw,
                                                                sram_1M_with_8_128K_bank_128_1r_1w_W_size, sram_1M_with_8_128K_bank_128_1r_1w_W_bw
                                                            ]
                                                            updated_accelerator = self.update_memory(self.accelerator, mem_config, self.node)
                                                            kwargs = pickle_deepcopy(self.kwargs)
                                                            kwargs["accelerator"] = updated_accelerator
                                                            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
                                                            for cme, extra_info in sub_stage.run():
                                                                energy_total = cme.energy_total
                                                                yield cme, extra_info
        elif(self.iterator_arch == "Tesla_NPU_like"):
            for reg_W1_size in [1,2,4]:
                for reg_W1_bw in [8,16]:
                    for reg_O4_size in [reg_W1_size*2, reg_W1_size*4]:
                        for reg_O4_bw in [8,16]:
                            for sram_1KB_256_1r_1w_I_size in [512,1024,2048]:
                                for sram_1KB_256_1r_1w_I_bw in [32,64,128,256]:
                                    for sram_1KB_256_1r_1w_W_size in [512,1024,2048]:
                                        for sram_1KB_256_1r_1w_W_bw in [32,64,128,256]:
                                            for sram_1M_with_8_128K_bank_128_1r_1w_A_size in [x*1024 for x in [64,128,256]]:
                                                for sram_1M_with_8_128K_bank_128_1r_1w_A_bw in [512,1024,2048]:
                                                    for sram_1M_with_8_128K_bank_128_1r_1w_W_size in [x*1024 for x in [64,128,256]]:
                                                        for sram_1M_with_8_128K_bank_128_1r_1w_W_bw in [512,1024,2048]:
                                                            mem_config = [
                                                                reg_W1_size, reg_W1_bw,
                                                                reg_O4_size, reg_O4_bw,
                                                                sram_1KB_256_1r_1w_I_size, sram_1KB_256_1r_1w_I_bw,
                                                                sram_1KB_256_1r_1w_W_size, sram_1KB_256_1r_1w_W_bw,
                                                                sram_1M_with_8_128K_bank_128_1r_1w_A_size, sram_1M_with_8_128K_bank_128_1r_1w_A_bw,
                                                                sram_1M_with_8_128K_bank_128_1r_1w_W_size, sram_1M_with_8_128K_bank_128_1r_1w_W_bw
                                                            ]
                                                            updated_accelerator = self.update_memory(self.accelerator, mem_config, self.node)
                                                            kwargs = pickle_deepcopy(self.kwargs)
                                                            kwargs["accelerator"] = updated_accelerator
                                                            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
                                                            for cme, extra_info in sub_stage.run():
                                                                energy_total = cme.energy_total
                                                                yield cme, extra_info
        else:
            print("Iterator Arch incorrect")  # error




