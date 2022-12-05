# -*- coding: utf-8 -*-
"""
Usage: html.py <dataset> <output> [--n=<n> --width=<width> --config=<path>]

Options:
--n=<n>                 Number of examples to output after shuffling. Defaults to all without shuffling.
--width=<width>         Width of the image in HTML [default: 400]. 
"""
from datasets import load_from_disk
from docopt import docopt
import json
from tqdm import tqdm
from ranx import Run


HTML_TEMPLATE = """<html>
<head>
    <link rel="stylesheet" href="./styles.css">
</head>
<table>
    {headers}
    {rows}
<tr>"""


def get_top_1(item, run):
    results = run.run[item['id']]
    top1 = next(iter(results))
    return int(top1)


def format_html(dataset_path, output, n=None, width=400, 
                passages_path=None, wiki_path=None, search_run=None, other_search_run=None):
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
            passage = passages[i]
            article = wiki[passage['index']]
            row['passage_url'] = article['url']
            row['passage_text'] = passage['passage']
        if other_search_run is not None:
            i = get_top_1(item, other_search_run)
            passage = passages[i]
            article = wiki[passage['index']]
            row['other_passage_url'] = article['url']
            row['other_passage_text'] = passage['passage']
        rows.append(row_template.format(**row))
    html_str = HTML_TEMPLATE.format(headers=headers, rows='\n'.join(rows))
    with open(output, 'wt') as file:
        file.write(html_str)
    
    
if __name__ == '__main__':
    args = docopt(__doc__)
    n = int(args['--n']) if args['--n'] is not None else None
    if args['--config'] is not None:
        with open(args['--config'], 'rt') as file:
            config = json.load(file)
    else:
        config = {}
    format_html(args['<dataset>'], args['<output>'], n=n, width=int(args['--width']), **config)