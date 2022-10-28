# -*- coding: utf-8 -*-

"""Usage: umap.py <dataset> <key> <output> [<config>]"""
from docopt import docopt
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool, ColumnDataSource, LinearColorMapper, ColorBar

from datasets import load_from_disk

import ranx

import urllib
from ..data.wiki import thumbnail_to_file_name, file_name_to_thumbnail


def reduce(embeddings, metric='cosine'):    
    reducer = umap.UMAP(metric=metric)
    reducer.fit(embeddings)
    reduced_embeddings = reducer.transform(embeddings)
    return reduced_embeddings


def fplot(reduced_embeddings, figsize=(20,20), alpha=0.5, s=5):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=alpha, s=s)
    return fig


def iplot(reduced_embeddings, dataset, urls, input_key='input', thumb_width=128, title='UMAP projection',
          plot_width=600, plot_height=600, tools=('pan, wheel_zoom, reset'),
          line_alpha=0.6, fill_alpha=0.6, size=4, metric=None, palette='Viridis256'):
    thumbnails = [file_name_to_thumbnail(thumbnail_to_file_name(urllib.parse.unquote(url)), thumb_width) for url in urls]
    df = pd.DataFrame(reduced_embeddings, columns=('x', 'y'))
    if input_key is None:
        df['text'] = ['']*len(dataset)
    else:
        df['text'] = dataset[input_key]
    df['image'] = thumbnails
    if metric is not None:
        df[metric] = dataset[metric]
        cmap = LinearColorMapper(palette=palette, low = df[metric].min(), high = df[metric].max())
        fill_color = {'field': metric, 'transform': cmap}
    else:
        fill_color = 'gray'
        cmap = None
    
    datasource = ColumnDataSource(df)
    
    fig = figure(title=title, plot_width=plot_width, plot_height=plot_height, tools=tools)
    fig.add_tools(HoverTool(tooltips="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
        </div>
        <div>
            <span style='font-size: 10px'>@text</span>
        </div>
    </div>
    """))
    fig.circle(
        'x',
        'y',
        source=datasource,
        line_alpha=line_alpha,
        fill_alpha=fill_alpha,
        size=size,
        fill_color=fill_color
    )
    if cmap is not None:
        cb = ColorBar(color_mapper = cmap)
        fig.add_layout(cb, 'right')
    return fig


def get_ranx_run(qrels_path, run_path, metric='mrr'):
    qrels = ranx.Qrels.from_file(qrels_path)
    run = ranx.Run.from_file(run_path)
    ranx.evaluate(qrels, run, metrics=metric)
    return run, metric

    
def main(dataset, key, output_path, image_kb=None, shard=None, url_key='url',
         reduce_kwargs={}, fplot_kwargs={}, iplot_kwargs={}, ranx_kwargs=None, face=False):
    output_path.mkdir(exist_ok=True)
    if shard is not None:
        dataset = dataset.shuffle(seed=0).shard(shard, 0)
    if face:
        dataset = dataset.filter(lambda x: x is not None, input_columns=key)
        embeddings = np.array([x[0] for x in dataset[key]])
    else:
        embeddings = np.array(dataset[key])
    # add features from the image KB
    if image_kb is not None:
        image_kb = load_from_disk(image_kb)
        print(image_kb)
        indices = dataset['index']
        print(len(indices))
        urls = image_kb.select(indices)[url_key]
        print(len(urls))
    else:
        urls = dataset[url_key]
    if ranx_kwargs is not None:
        ranx_run, metric = get_ranx_run(**ranx_kwargs)   
        dataset=dataset.map(lambda item: {metric: ranx_run.scores[metric][item["id"]]})
    else:
        ranx_run, metric = None, None
    reduced_embeddings = reduce(embeddings, **reduce_kwargs)
    ffig = fplot(reduced_embeddings, **fplot_kwargs)
    ffig.savefig(output_path/f"umap_{key}.png")
    ifig = iplot(reduced_embeddings, dataset, urls=urls, metric=metric, **iplot_kwargs)
    output_file(output_path/f"umap_{key}.html")
    save(ifig)


if __name__ == '__main__':
    args = docopt(__doc__)
    dataset = load_from_disk(args['<dataset>'])
    config_path = args['<config>']
    if config_path is not None:
        with open(config_path, 'rt') as file:
            config = json.load(file)
    else:
        config = {}
    output_path = Path(args['<output>'])
    main(dataset, args['<key>'], output_path, **config)