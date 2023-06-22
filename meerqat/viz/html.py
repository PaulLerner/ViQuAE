# -*- coding: utf-8 -*-
"""
usage: html.py [-h] [--config CONFIG] [--print_config [={comments,skip_null,skip_default}+]] 
[--n N] [--width WIDTH] [--passages_path PASSAGES_PATH] [--wiki_path WIKI_PATH] 
[--search_run SEARCH_RUN] [--other_search_run OTHER_SEARCH_RUN] dataset_path output 

options:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config [={comments,skip_null,skip_default}+]
                        Print the configuration after applying all other arguments and exit.

Visualize dataset in HTML:
  dataset_path          (required, type: str)
  output                (required, type: str)
  --n N                 Number of examples to output after shuffling. Defaults to all without
                        shuffling. (type: Optional[int], default: null)
  --width WIDTH         Width of the image in HTML [default: 400]. (type: int, default: 400)
  --passages_path PASSAGES_PATH
                        (type: Optional[str], default: null)
  --wiki_path WIKI_PATH
                        (type: Optional[str], default: null)
  --search_run SEARCH_RUN
                        (type: Optional[str], default: null)
  --other_search_run OTHER_SEARCH_RUN
                        (type: Optional[str], default: null)
"""
import json
from tqdm import tqdm
from jsonargparse import CLI

from datasets import load_from_disk
from ranx import Run


HTML_TEMPLATE = """<html>
<head>
    <link rel="stylesheet" href="./styles.css">
</head>
<table>
    {headers}
    {rows}
</table>
</html>
"""


def get_top_1(item, run):
    results = run.run[item['id']]
    top1 = next(iter(results))
    return int(top1)


def get_url_and_text(i, wiki, passages):
    if passages is not None:
        passage = passages[i]
        article = wiki[passage['index']]
        return article['url'], passage['passage']
    else:
        article = wiki[i]
        return article['url'], article['wikipedia_title']
    
    
def format_html(
        dataset_path: str, 
        output: str, 
        n: int = None, 
        width: int = 400, 
        passages_path: str = None,
        wiki_path: str = None, 
        search_run: str = None, 
        other_search_run: str = None
    ):
    """
    Visualize dataset in HTML
    
    Parameters
    ----------
    dataset_path: str
    output: str
    n: int
        Number of examples to output after shuffling. 
        Defaults to all without shuffling.
    width: int 
        Width of the image in HTML [default: 400].
    passages_path: str
    wiki_path: str
    search_run: str
    other_search_run: str
    """
    # complete template according to parameters
    if search_run is not None:
        search_run = Run.from_file(search_run)
        search_headers = f"""<th>Visual</th>
        <th>Passage ({search_run.name})</th>"""
        search_row_template = """<td><img src="{passage_url}" width="{width}"></td>
        <td>{passage_text}</td>"""
    else:
        search_headers, search_row_template = "", ""
    if other_search_run is not None:
        other_search_run = Run.from_file(other_search_run)
        other_search_headers = f"""<th>Other Visual</th>
        <th>Passage ({other_search_run.name})</th>"""
        other_search_row_template = """<td><img src="{other_passage_url}" width="{width}"></td>
        <td>{other_passage_text}</td>"""
    else:
        other_search_headers, other_search_row_template = "", ""
    headers = """<tr>
        <th>Visual</th>
        <th>Question</th>
        <th>Answer</th>
        %s
        %s
    </tr>""" % (search_headers, other_search_headers)
    row_template = """<tr>
        <td><img src="{url}" width="{width}"></td>
        <td>{question}</td>
        <td>{answer}</td>
        %s
        %s
    </tr>""" % (search_row_template, other_search_row_template)
    
    # load data
    dataset = load_from_disk(dataset_path)
    if n is not None:
        dataset = dataset.shuffle().select(range(n))
    if passages_path is not None:
        passages = load_from_disk(passages_path)
    else:
        passages = None
    if wiki_path is not None:
        wiki = load_from_disk(wiki_path)
    
    # do the actual formatting
    rows = []
    for item in tqdm(dataset):
        row = dict(
                url=item['url'],
                width=width,
                question=item['input'],
                answer=item['output']['original_answer']
        )
        if search_run is not None:
            i = get_top_1(item, search_run)
            row['passage_url'], row['passage_text'] = get_url_and_text(i, wiki, passages)
        if other_search_run is not None:
            i = get_top_1(item, other_search_run)
            row['other_passage_url'], row['other_passage_text'] = get_url_and_text(i, wiki, passages)
        rows.append(row_template.format(**row))
    html_str = HTML_TEMPLATE.format(headers=headers, rows='\n'.join(rows))
    with open(output, 'wt') as file:
        file.write(html_str)
    
    
if __name__ == '__main__':
    CLI(format_html)