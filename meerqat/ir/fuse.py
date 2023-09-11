# -*- coding: utf-8 -*-
"""
usage: fuse.py [-h] [--config CONFIG] [--print_config [={comments,skip_null,skip_default}+]] [--qrels QRELS] [--runs RUNS] [--norm NORM] [--method METHOD] [--output OUTPUT] [--defmin DEFMIN]
               {fit,test} ...

options:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config [={comments,skip_null,skip_default}+]
                        Print the configuration after applying all other arguments and exit.

Optimize fusion using ranx:
  --qrels QRELS, --qrels+ QRELS
                        (type: Union[str, List[str], null], default: null)
  --runs RUNS, --runs+ RUNS
                        (type: Optional[List[str]], default: null)
  --norm NORM, --norm+ NORM
                        (type: Union[str, null, List[Optional[str]]], default: zmuv)
  --method METHOD, --method+ METHOD
                        (type: Union[str, null, List[Optional[str]]], default: wsum)
  --output OUTPUT       (type: Optional[str], default: null)
  --defmin DEFMIN, --defmin+ DEFMIN
                        (type: Union[bool, null, List[Optional[bool]]], default: False)

subcommands:
  For more details of each subcommand add it as argument followed by --help.

  {fit,test}
    fit                 Finds best parameters
    test                Applies best parameters
"""
from jsonargparse import CLI
from typing import List, Optional, Union
import yaml
from pathlib import Path
import json

import numpy as np

from numba import njit, prange, types
from numba.typed import List as TypedList, Dict as TypedDict
from ranx import fuse, optimize_fusion, Run, Qrels, evaluate

from ..data.utils import to_latex
from .metrics import fuse_qrels


####################################
# copied from ranx because private #
####################################

@njit(cache=True)
def create_empty_results_dict():
    return TypedDict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )

@njit(cache=True)
def create_empty_results_dict_list(length):
    return TypedList([create_empty_results_dict() for _ in range(length)])

@njit(cache=True)
def convert_results_dict_list_to_run(q_ids, results_dict_list):
    combined_run = TypedDict()

    for i, q_id in enumerate(q_ids):
        combined_run[q_id] = results_dict_list[i]

    return combined_run


@njit(cache=True)
def extract_scores(results):
    """Extract the scores from a given results dictionary."""
    scores = np.empty((len(results)))
    for i, v in enumerate(results.values()):
        scores[i] = v
    return scores


################
# custom norms #
################
    
@njit(cache=True)
def _gzmuv_norm(results, mean_score, stdev_score):
    """Apply `gzmuv norm` to a given results dictionary."""
    denominator = max(stdev_score, 1e-9)

    normalized_results = create_empty_results_dict()
    for doc_id in results.keys():
        normalized_results[doc_id] = (results[doc_id] - mean_score) / (
            denominator
        )

    return normalized_results


@njit(cache=True, parallel=True)
def _gzmuv_norm_parallel(run):
    """Apply `zmuv norm` to a each results dictionary of a run in parallel."""
    q_ids = TypedList(run.keys())

    normalized_run = create_empty_results_dict_list(len(q_ids))
    # FIXME getting error np.concatenate(): expecting a non-empty tuple of arrays, got ListType[array(float64, 1d, C)]
    #    results_lengths = [len(results) for results in run.values()]
    #    all_scores = TypedList([np.empty(l) for l in results_lengths])
    #    for i in prange(len(q_ids)):
    #        all_scores[i] = extract_scores(run[q_ids[i]])
    #    all_scores = np.concatenate(list(all_scores))
    
    # not very numba-esque hack
    all_scores = np.array([v for results in run.values() for v in results.values()])
    mean_score = np.mean(all_scores)
    stdev_score = np.std(all_scores)
    
    for i in prange(len(q_ids)):
        normalized_run[i] = _gzmuv_norm(run[q_ids[i]], mean_score, stdev_score)

    return convert_results_dict_list_to_run(q_ids, normalized_run)


def gzmuv_norm(run):
    """Apply `gzmuv norm` to a run."""
    normalized_run = Run()
    normalized_run.name = run.name
    normalized_run.run = _gzmuv_norm_parallel(run.run)
    return normalized_run


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
            if len(results) == 0:
                continue
            minimum = min(results.values())
            for d_id in all_documents[q_id]:
                results.setdefault(d_id, minimum)
    
    return runs
    
                
NORMS = {
    "gzmuv": gzmuv_norm
}


##################
# Main class/CLI #
##################


class Fusion:
    """Optimize fusion using ranx"""
    def __init__(
        self,
        qrels: Union[str, Path, Qrels, List[str]] = None,
        runs: Union[List[str], List[Run]] = None,        
        norm: Union[Optional[str], List[Optional[str]]] = "zmuv",
        method: Union[Optional[str], List[Optional[str]]] = "wsum",
        output: Optional[str] = None,
        defmin: Optional[bool] = False
    ):
        if isinstance(qrels, Qrels) or qrels is None:
            self.qrels = qrels
        elif isinstance(qrels, (str, Path)):
            self.qrels = Qrels.from_file(qrels)
        else:
            self.qrels = fuse_qrels(qrels)
        if isinstance(runs[0], Run):
            self.runs = runs
        else:
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
        norms = [self.norm] if self.norm is None or isinstance(self.norm, str) else self.norm 
        methods = [self.method] if self.method is None or isinstance(self.method, str) else self.method 
        for norm in norms:
            # custom norm: do it as a pre-processing and disable ranx norm
            if norm in NORMS:
                runs = [NORMS[norm](run) for run in self.runs]
                norm_for_ranx = None
            else:
                runs = self.runs
                norm_for_ranx = norm
            for method in methods:
                best_params, report = optimize_fusion(
                    qrels=self.qrels,
                    runs=runs,
                    norm=norm_for_ranx,
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
        if metrics is None:
            metrics = ["mrr@100", "precision@1", "precision@20", "hit_rate@20"]
        # custom norm: do it as a pre-processing and disable ranx norm
        if self.norm in NORMS:
            self.runs = [NORMS[self.norm](run) for run in self.runs]
            self.norm = None
        combined_run = fuse(
            runs=self.runs,
            norm=self.norm,       
            method=self.method,        
            params=best_params
        )
        if self.output is not None:
            combined_run.save(self.output/"test_run.json")
        if metrics is not None:
            print(to_latex(evaluate(self.qrels, combined_run, metrics)))
        return combined_run

    
if __name__ == '__main__':
    CLI(Fusion)