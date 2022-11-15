# -*- coding: utf-8 -*-

from jsonargparse import CLI
from typing import List
import yaml

from ranx import fuse, optimize_fusion, Run, Qrels, evaluate


class Main:
    """Optimize fusion using ranx"""
    def __init__(
        self,
        qrels: str = None,
        runs: List[str] = None,        
        norm: str = "zmuv",
        method: str = "wsum",
        output: str = None
    ):
        self.qrels = Qrels.from_file(qrels)
        self.runs = [Run.from_file(run) for run in runs]
        self.norm = norm
        self.method = method
        self.output = output
    
    def fit(self, metric: str = "mrr@100"):
        """Finds best parameters"""
        best_params = optimize_fusion(
            qrels=self.qrels,
            runs=self.runs,
            norm=self.norm,
            method=self.method,
            metric=metric
        )
        print(best_params)
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
            print(evaluate(self.qrels, combined_run, metrics))

    
if __name__ == '__main__':
    CLI(Main)