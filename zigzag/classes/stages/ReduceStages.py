import logging
import pandas as pd
from datetime import datetime

from typing import Generator, Callable, List, Tuple, Any
from zigzag.classes.stages.Stage import Stage
from zigzag.classes.cost_model.cost_model import CostModelEvaluation
logger = logging.getLogger(__name__)


class MinimalEnergyStage(Stage):
    """
    Class that keeps yields only the cost model evaluation that has minimal energy of all cost model evaluations
    generated by it's substages created by list_of_callables
    """
    def __init__(self, list_of_callables, *, reduce_minimal_keep_others=False, **kwargs):
        """
        Initialize the compare stage.
        """
        super().__init__(list_of_callables, **kwargs)
        self.best_cme = None
        # Visualization stuff
        self.energies = []
        self.keep_others = reduce_minimal_keep_others


    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        """
        Run the compare stage by comparing a new cost model output with the current best found result.
        """
        sub_list_of_callables = self.list_of_callables[1:]
        substage = self.list_of_callables[0](sub_list_of_callables, **self.kwargs)

        other_cmes = []
        for cme, extra_info in substage.run():
            self.energies.append(cme.energy_total)
            if self.best_cme is None or cme.energy_total < self.best_cme.energy_total or (cme.energy_total == self.best_cme.energy_total and cme.latency_total2 < self.best_cme.latency_total2):
                self.best_cme = cme
            if self.keep_others:
                other_cmes.append((cme, extra_info))
        yield (self.best_cme, other_cmes)


class MinimalLatencyStage(Stage):
    """
    Class that keeps yields only the cost model evaluation that has minimal latency of all cost model evaluations
    generated by it's substages created by list_of_callables
    """
    def __init__(self, list_of_callables,*, reduce_minimal_keep_others=False, **kwargs):
        """
        Initialize the compare stage.
        """
        super().__init__(list_of_callables, **kwargs)
        self.best_cme = None
        self.keep_others = reduce_minimal_keep_others


    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        """
        Run the compare stage by comparing a new cost model output with the current best found result.
        """
        sub_list_of_callables = self.list_of_callables[1:]
        substage = self.list_of_callables[0](sub_list_of_callables, **self.kwargs)

        other_cmes = []
        for cme, extra_info in substage.run():
            if self.best_cme is None or cme.latency_total2 < self.best_cme.latency_total2 or (cme.latency_total2 == self.best_cme.latency_total2 and cme.energy_total < self.best_cme.energy_total):
                self.best_cme = cme
            if self.keep_others:
                other_cmes.append((cme, extra_info))
        yield (self.best_cme, other_cmes)

class MinimalEDPStage(Stage):
    """
    Class that keeps yields only the cost model evaluation that has minimal EDP of all cost model evaluations
    generated by it's substages created by list_of_callables
    """
    def __init__(self, list_of_callables,*, reduce_minimal_keep_others=False, **kwargs):
        """
        Initialize the compare stage.
        """
        super().__init__(list_of_callables, **kwargs)
        self.best_cme = None
        self.keep_others = reduce_minimal_keep_others


    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        """
        Run the compare stage by comparing a new cost model output with the current best found result.
        """
        sub_list_of_callables = self.list_of_callables[1:]
        substage = self.list_of_callables[0](sub_list_of_callables, **self.kwargs)

        other_cmes = []
        for cme, extra_info in substage.run():
            if self.best_cme is None or cme.latency_total2*cme.energy_total < self.best_cme.latency_total2*self.best_cme.energy_total:
                self.best_cme = cme
            if self.keep_others:
                other_cmes.append((cme, extra_info))
        yield (self.best_cme, other_cmes)

class SumStage(Stage):
    """
    Class that keeps yields only the sum of all cost model evaluations generated by its
    substages created by list_of_callables
    """
    def __init__(self, list_of_callables, **kwargs):
        """
        Initialize the compare stage.
        """
        super().__init__(list_of_callables, **kwargs)
        self.total_cme = None


    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        """
        Run the compare stage by comparing a new cost model output with the current best found result.
        """
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

        all_cmes = []
        for cme, extra_info in substage.run():
            if self.total_cme is None:
                self.total_cme = cme
            else:
                self.total_cme += cme
            all_cmes.append((cme, extra_info))
        yield self.total_cme, all_cmes

class ListifyStage(Stage):
    """Class yields all the cost model evaluations yielded by its substages as a single list instead of as a generator."""
    def __init__(self, list_of_callables, **kwargs):
        """
        Initialize the compare stage.
        """
        super().__init__(list_of_callables, **kwargs)
        self.list = []


    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        """
        Run the compare stage by comparing a new cost model output with the current best found result.
        """
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

        for cme, extra_info in substage.run():
            self.list.append((cme, extra_info))
        yield self.list, None
        

class csvStage(Stage):
    """Class saves to csv."""
    def __init__(self, list_of_callables, **kwargs):
        """
        Initialize the compare stage.
        """
        super().__init__(list_of_callables, **kwargs)
        self.list = []


    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        """
        write cmes to csv.
        """
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
         
        rows_list = []
        count = 0
        for cme, extra_info in substage.run():
            row = {'energy': cme.energy_total,
                   'latency': cme.latency_total2,
                   'hw': cme.cfg}
            count += 1
            rows_list.append(row)
            if (count % 10 == 0):
                # put this here since this each iteration takes significant time
                # in case of error there would still be some data saved
                df = pd.DataFrame(rows_list)
                str_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
                print("Updating CSV ...")
                df.to_csv('./outputs/cme_list-' + str(cme.cfg[0]) + '-' + str_datetime + '.csv')
            self.list.append((cme, extra_info))
        # put this here since this each iteration takes significant time
        # in case of error there would still be some data saved
        df = pd.DataFrame(rows_list)
        str_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        print("Updating CSV ...")
        df.to_csv('./outputs/cme_list-' + str(cme.cfg[0]) + '-' + str_datetime + '.csv')
        yield self.list, None
        
