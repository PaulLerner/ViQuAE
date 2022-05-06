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
from bokeh.models import HoverTool, ColumnDataSource

from datasets import load_from_disk

from meerqat.data.wiki import thumbnail_to_file_name, file_name_to_thumbnail


def reduce(embeddings, metric='cosine'):    
    reducer = umap.UMAP(metric=metric)
    reducer.fit(embeddings)
    reduced_embeddings = reducer.transform(embeddings)
    return reduced_embeddings


def fplot(reduced_embeddings, figsize=(20,20), alpha=0.5, s=5):
    output_path.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=alpha, s=s)
    return fig


def iplot(reduced_embeddings, dataset, urls, input_key='input', thumb_width=128, title='UMAP projection',
          plot_width=600, plot_height=600, tools=('pan, wheel_zoom, reset'),
          line_alpha=0.6, fill_alpha=0.6, size=4):
    thumbnails = [file_name_to_thumbnail(thumbnail_to_file_name(url), thumb_width) for url in urls]
    df = pd.DataFrame(reduced_embeddings, columns=('x', 'y'))
    df['text'] = dataset[input_key]
    df['image'] = thumbnails
    
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
        size=size
    )
    return fig

    
def main(dataset, key, output_path, image_kb=None, shard=None, reduce_kwargs={}, fplot_kwargs={}, iplot_kwargs={}):
    if shard is not None:
        dataset = dataset.shuffle(seed=0).shard(shard, 0)
    print(dataset)
    # add features from the image KB
    if image_kb is not None:
        image_kb = load_from_disk(image_kb)
        print(image_kb)
        indices = dataset['index']
        print(len(indices))
        urls = image_kb.select(indices)['url']
        print(len(urls))
    else:
        urls = dataset['url']
    print(dataset)
    embeddings = np.array(dataset[key])
    print(embeddings.shape)
    reduced_embeddings = reduce(embeddings, **reduce_kwargs)
    print(reduced_embeddings.shape)
    ffig = fplot(reduced_embeddings, **fplot_kwargs)
    ffig.savefig(output_path/f"umap_{key}.png")
    ifig = iplot(reduced_embeddings, dataset, urls=urls, **iplot_kwargs)
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