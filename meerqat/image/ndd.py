# -*- coding: utf-8 -*-
"""
usage: ndd.py [-h] [--config CONFIG] [--print_config[=flags]]
              {list,show} ...

Set of tools to bridge meerqat and compute_dlphash and search_direct_binary

optional arguments:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config[=flags]
                        Print the configuration after applying all other
                        arguments and exit. The optional flags are one or more
                        keywords separated by comma which modify the output.
                        The supported flags are: comments, skip_default,
                        skip_null.

subcommands:
  For more details of each subcommand add it as argument followed by --help.

  {list,show}
    list                Save all paths to the dataset images in an external
                        file
    show                Parses search_direct_binary output to show results
"""
from jsonargparse import CLI
from jsonargparse.typing import register_type, Path_fr
from pathlib import Path

import pandas as pd

from datasets import load_from_disk

from ..data.loading import IMAGE_PATH


register_type(Path, type_check=lambda v, t: isinstance(v, t))


class NearDuplicateDetection:
    def list(self, dataset_path: Path):
        """Save all paths to the dataset images in an external file"""
        dataset = load_from_disk(dataset_path)
        images = set(str(IMAGE_PATH/image) for image in dataset['image'])
        with open(dataset_path/'images.lst', 'wt') as file:
            file.write('\n'.join(images))
            
    def show(self, results: Path_fr, ref: Path_fr, queries: Path_fr):
        """Parses search_direct_binary output to show results"""
        results = pd.read_csv(results, sep=' ', header=None)
        ref = pd.read_csv(ref, names=['ref']).ref
        queries = pd.read_csv(queries, names=['query'])
        results = pd.concat((queries, results), axis=1)
        print(results)
        run = {}
        for _, (query, *rest) in results.iterrows():
            run[query] = {}
            for i in range(0, len(rest), 2):
                # results are stored like "<top1_index> <top1_score> <top2_index> <top2_score> …"
                run[query][ref[rest[i]]] = rest[i+1]
        print(run)      
        
        
if __name__ == '__main__':
    CLI(
        NearDuplicateDetection, 
        description="Set of tools to bridge meerqat and compute_dlphash and search_direct_binary"
    )    
