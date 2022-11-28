# -*- coding: utf-8 -*-

from jsonargparse import CLI
from typing import List, Optional
import yaml

from ranx import fuse, optimize_fusion, Run, Qrels, evaluate

from ..data.utils import to_latex


def default_minimum(runs):
    # union results
    all_documents = {}
    for run in runs:
        for q_id, results in run.run.items():
            all_documents.setdefault(q_id, set())
            all_documents[q_id] |= results.keys()
            
    # set default-minimum in runs
    for run in runs:
        for q_id, results in run.run.items():
            minimum = min(results.values())
            for d_id in all_documents[q_id]:
                results.setdefault(d_id, minimum)
    
    return runs
                    

class Main:
    """Optimize fusion using ranx"""
    def __init__(
        self,
        qrels: str = None,
        runs: List[str] = None,        
        norm: Optional[str] = "zmuv",
        method: Optional[str] = "wsum",
        output: str = None,
        defmin: Optional[bool] = False
    ):
        self.qrels = Qrels.from_file(qrels)
        self.runs = [Run.from_file(run) for run in runs]
        if defmin:
            self.runs = default_minimum(self.runs)
        self.norm = norm
        self.method = method
        self.output = output
    
    def fit(self, metric: str = "mrr@100"):
        """Finds best parameters"""
        best_params, report = optimize_fusion(
            qrels=self.qrels,
            runs=self.runs,
            norm=self.norm,
            method=self.method,
            metric=metric,
            return_optimization_report=True
        )
        print(best_params)
        print(report)
        if self.output is not None:
            with open(self.output, 'wt') as file:
                yaml.dump(best_params, file)
    
    def test(self, best_params: dict, metrics: List[str] = None):
        """Applies best parameters"""
        combined_run = fuse(
            runs=self.runs,
            norm=self.norm,       
            method=self.method,        
            params=best_params
        )
        combined_run.save(self.output)
        if metrics is not None:
            print(to_latex(evaluate(self.qrels, combined_run, metrics)))

    
if __name__ == '__main__':
    CLI(Main)