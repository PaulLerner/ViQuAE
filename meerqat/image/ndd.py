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
import json

import random

import pandas as pd
import seaborn as sns

from datasets import load_from_disk

from ..data.loading import IMAGE_PATH

random.seed(0)
register_type(Path, type_check=lambda v, t: isinstance(v, t))

HTML_TEMPLATE = """<html>
<head>
    <link rel="stylesheet" href="./styles.css">
</head>
<table>
    <tr>
        <th>Query</th>
        {top_headers}
    </tr>
    {rows}
</table>
</html>
"""
QUERY_TEMPLATE = """<td><img src="{url}" width="400"></td>"""
RESULT_TEMPLATE = """<td><img src="{url}" width="400"><p>{score}</p></td>"""

VALID_ENCODING = {'jpeg', 'jpg', 'png'}


class NearDuplicateDetection:    
    def list(self, dataset_path: Path):
        """Save all paths to the dataset images in an external file"""
        dataset = load_from_disk(dataset_path)
        def add_image_path(image, images):
            if image.split('.')[-1].lower() in VALID_ENCODING:
                images.add(str(IMAGE_PATH/image))
        images = set()
        dataset.map(add_image_path, input_columns='image', fn_kwargs=dict(images=images))
        with open(dataset_path/'images.lst', 'wt') as file:
            file.write('\n'.join(images))
            
    def show(self, results: Path_fr, ref: Path_fr, queries: Path_fr, output: Path, 
             n: int = 50, k: int = 5, minimum: float = None, maximum: float = None):
        """Parses search_direct_binary output to show results"""
        results = pd.read_csv(results, sep=' ', header=None)
        K = (results.shape[1]-1)//2
        scores = pd.concat([results[i*2+1] for i in range(K)])
        print(scores.describe())
        fig = sns.displot(data=scores, discrete=True)
        fig.savefig(output/f"scores_distribution.png")
        ref = pd.read_csv(ref, sep=' ', names=['ref']).ref
        queries = pd.read_csv(queries, sep=' ', names=['query'])
        results = pd.concat((queries, results), axis=1)        
        print(results)
        run = {}
        for _, (query, *rest) in results.iterrows():
            run[query] = {}
            # -1 because there’s an extra NaN at the end
            for i in range(0, len(rest)-1, 2):
                # results are stored like "<top1_index> <top1_score> <top2_index> <top2_score> …"
                run[query][ref[rest[i]]] = rest[i+1]
        with open(output/"run.json", 'wt') as file:
            json.dump(run, file)
        queries = list(run.keys())
        random.shuffle(queries)
        rows = []
        for query in queries:
            result = run[query]
            row = []
            for path, score in result.items():
                if minimum is not None and score < minimum:
                    continue
                if maximum is not None and score > maximum:
                    continue
                row.append(RESULT_TEMPLATE.format(url=path, score=score))
                if len(row) >= k:
                    break
            if row:
                rows.append("<tr>"+"\n".join([QUERY_TEMPLATE.format(url=query)] + row)+"</tr>")
            if len(rows) >= n:
                break
        top_headers = "\n".join(f"<th>Top-{i+1}</th>" for i in range(k))
        html = HTML_TEMPLATE.format(top_headers=top_headers, rows = "\n".join(rows))
        with open(output/f"{n}_top_{k}_min_{minimum}_max_{maximum}.html", 'wt') as file:
            file.write(html)
                
        
if __name__ == '__main__':
    CLI(
        NearDuplicateDetection, 
        description="Set of tools to bridge meerqat and compute_dlphash and search_direct_binary"
    )    
