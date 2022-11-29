# -*- coding: utf-8 -*-

from jsonargparse import CLI
from typing import List, Optional, Union
import yaml
from pathlib import Path
import json

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
        qrels: Union[str, List[str]] = None,
        runs: List[str] = None,        
        norm: Union[Optional[str], List[Optional[str]]] = "zmuv",
        method: Union[Optional[str], List[Optional[str]]] = "wsum",
        output: Optional[str] = None,
        defmin: Optional[bool] = False
    ):
        # TODO if list: merge using metrics.fuse_qrels
        self.qrels = Qrels.from_file(qrels)
        self.runs = [Run.from_file(run) for run in runs]
        if defmin:
            self.runs = default_minimum(self.runs)
        self.norm = norm
        self.method = method
        if output is not None:
            # FIXME: use jsonargparse Path instead of pathlib? https://github.com/omni-us/jsonargparse/issues/94
            output = Path(output)
            output.mkdir(exist_ok=True)
        self.output = output
    
    def fit(self, metric: str = "mrr@100"):
        """Finds best parameters"""
        norms = [self.norm] if isinstance(self.norm, str) else self.norm 
        methods = [self.method] if isinstance(self.method, str) else self.method 
        for norm in norms:
            for method in methods:
                best_params, report = optimize_fusion(
                    qrels=self.qrels,
                    runs=self.runs,
                    norm=norm,
                    method=method,
                    metric=metric,
                    return_optimization_report=True
                )
                print(f"Norm: {norm}, Method: {method}. Best parameters: {best_params}.\n{report}")
                if self.output is not None:
                    with open(self.output/f"{norm}_{method}_best_params.yaml", 'wt') as file:
                        yaml.dump(json.loads(json.dumps(best_params)), file)
    
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