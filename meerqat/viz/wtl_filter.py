# coding: utf-8
"""
usage: wtl_filter.py [-h] --runA RUNA [--runB RUNB]
                     [--filter {win,loose,intersection,union,nunion,nintersection}] [--kA KA]
                     [--kB KB]
                     <dataset> <output> <qrels>

Filter a dataset for questions according to runs A and B and filter option. For example, find
questions where A wins over B. By default (kA=kB=1), winning means that A has relevant result in
top-1, while B top-1 is irrelevant. However, you can change the definition, e.g. kA=1, kB=100,
means that A wins if it has a relevant result in top-1 and B has only irrelevant in its top-100.

positional arguments:
  <dataset>             Path of the dataset to filter.
  <output>              Path to save the filtered dataset.
  <qrels>               Path to the qrels.

options:
  -h, --help            show this help message and exit
  --runA RUNA           Path to run A. (default: None)
  --runB RUNB           Path to run B. Defaults: run A always win when its right. (default: None)
  --filter {win,loose,intersection,union,nunion,nintersection}
                        Save when A wins over B (default), looses over B, both wins (intersection),
                        either win (union), both loose (nunion), or one looses (nintersection)
                        (default: win)
  --kA KA               Judge A on its top-K. (default: 1)
  --kB KB               Judge B on its top-K. (default: 1)
"""

import argparse
import ranx
from datasets import load_from_disk

def get_wins(run, topk=1):
    return {k for k, v in run.scores[f'hit_rate@{topk}'].items() if v > 0}
    

def main(dataset, output, qrels, runA, runB=None, filter='win', kA=1, kB=1):
    d = load_from_disk(dataset)
    
    qrels = ranx.Qrels.from_file(qrels)
    runA = ranx.Run.from_file(runA)
    evaluateA = ranx.evaluate(qrels=qrels, run=runA, metrics=f'hit_rate@{kA}')
    runA_wins = get_wins(runA, topk=kA)
    runA.name = "run A" if runA.name is None else runA.name
    print(runA.name, f'hit_rate@{kA}:', evaluateA, f"wins: {len(runA_wins)}")
    if runB is not None:
        runB = ranx.Run.from_file(runB) 
        evaluateB = ranx.evaluate(qrels=qrels, run=runB, metrics=f'hit_rate@{kB}')
        runB_wins = get_wins(runB, topk=kB)
        runB.name = "run B" if runB.name is None else runB.name
        print(runB.name, f'hit_rate@{kB}:', evaluateB, f"wins: {len(runB_wins)}")
        intersection, union = runA_wins&runB_wins, runA_wins|runB_wins   
        print(f"intersection: {len(intersection)}, union: {len(union)}")
        print(f"A over B: {len(runA_wins-runB_wins)}, B over A: {len(runB_wins-runA_wins)}")  
        switch_filter = {
            'win': lambda id_: id_ in runA_wins-runB_wins,
            'loose': lambda id_: id_ not in runB_wins-runA_wins,
            'intersection': lambda id_: id_ in intersection,
            'union': lambda id_: id_ in union,
            'nunion': lambda id_: id_ not in union, 
            'nintersection': lambda id_: id_ not in intersection
        }
        filtered = d.filter(switch_filter[filter], input_columns='id')
    else:   
        switch_filter = {
            'win': lambda id_: id_ in runA_wins,
            'loose': lambda id_: id_ not in runA_wins
        }
        filtered = d.filter(switch_filter[filter], input_columns='id')
    print(filtered)
    filtered.save_to_disk(output)
        
    
if __name__ =='__main__':
    description = """
    Filter a dataset for questions according to runs A and B and filter option.
    For example, find questions where A wins over B.
    By default (kA=kB=1), winning means that A has relevant result in top-1, while B top-1 is irrelevant.
    However, you can change the definition, e.g. kA=1, kB=100, 
    means that A wins if it has a relevant result in top-1 and B has only irrelevant in its top-100.
    """
    parser = argparse.ArgumentParser(
        description=description, 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('dataset', metavar='<dataset>', type=str, help='Path of the dataset to filter.')
    parser.add_argument('output', metavar='<output>', type=str, help='Path to save the filtered dataset.')
    parser.add_argument('qrels', metavar='<qrels>', type=str, help='Path to the qrels.')
    parser.add_argument('--runA', type=str, help='Path to run A.', required=True)
    parser.add_argument('--runB', type=str, help='Path to run B. Defaults: run A always win when its right.', required=False, default=None)
    parser.add_argument('--filter', type=str, choices=['win', 'loose', 'intersection', 'union', 'nunion', 'nintersection'], required=False, default='win', 
        help="Save when A wins over B (default), looses over B, both wins (intersection), either win (union), both loose (nunion), or one looses (nintersection)")
    parser.add_argument('--kA', type=int, help='Judge A on its top-K.', required=False, default=1)
    parser.add_argument('--kB', type=int, help='Judge B on its top-K.', required=False, default=1)
    args = parser.parse_args()
    main(**vars(args))
