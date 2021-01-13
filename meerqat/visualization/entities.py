# coding: utf-8
"""Usage:
entities.py <subset>
"""

from docopt import docopt
import json
from pathlib import Path
from tabulate import tabulate

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from meerqat.data.loading import DATA_ROOT_PATH
from meerqat.visualization.utils import simple_stats


def count_entities(entities, distinct=False):
    """Note this counts labels and not QIDs

    Parameters
    ----------
    entities: List[dict]
    distinct: bool
        Whether to count distinct entities or # of questions per entity
        e.g. if we have 2 questions about Barack Obama, 'distinct' counts one human
        and '!distinct' counts 2
        This has no effect on "depiction_dist"
        which counts the # of depictions per (distinct) entity

    Returns
    -------
    counters: dict[Counter]
    """
    counters = {
        "commons": Counter(),
        "image": Counter(),
        "instanceof": Counter(),
        "gender": Counter(),
        "occupation": Counter(),
        "depictions": Counter(),
        "depiction_dist": []
    }
    for entity in entities.values():
        n = 1 if distinct else entity["n_questions"]

        # is commons category, image or depictions available ?
        for key in ["commons", "image", "depictions"]:
            counters[key][bool(entity.get(key))] += n

        # how many depictions per entity ?
        counters["depiction_dist"].append(len(entity.get("depictions", [])))

        # does it have a gender ? if yes, which one ?
        genderLabel = entity.get('genderLabel')
        if genderLabel:
            counters["gender"][genderLabel["value"]] += n

        # else count all available values
        for key in ["instanceof", "occupation"]:
            if key not in entity:
                continue
            for item in entity[key].values():
                counters[key][item["label"]["value"]] += n
    return counters


def visualize_entities(counters, path=Path.cwd(), subset="meerqat"):
    # pie-plot counters with lot of values
    for key in ["instanceof", "occupation"]:
        counter = counters[key]

        # keep only the last decile for a better readability
        values = np.array(list(counter.values()))
        labels = np.array(list(counter.keys()))
        deciles = np.quantile(values, np.arange(0., 1.1, 0.1))
        where = values > deciles[-2]
        filtered_values = np.concatenate((values[where], values[~where].sum(keepdims=True)))
        filtered_labels = np.concatenate((labels[where], ["other"]))

        # plot and save figure
        plt.figure(figsize=(16, 16))
        title = f"Distribution of {key} in {subset}"
        plt.title(title)
        plt.pie(filtered_values, labels=filtered_labels)
        output = path / title.replace(" ", "_")
        plt.savefig(output)
        plt.close()
        print(f"Successfully saved {output}")

    # barplot distributions and print some stats
    for key in ["depiction_dist"]:
        counter = counters[key]

        # print some stats
        print(simple_stats(counter, tablefmt="latex"))

        # barplot
        plt.figure(figsize=(16, 16))
        title = f"Distribution of {key} in {subset}"
        plt.title(title)
        plt.hist(counter, bins=50, density=False)
        output = path / title.replace(" ", "_")
        plt.savefig(output)
        plt.close()
        print(f"Successfully saved {output}")

    # print statistics for counters with fue values
    for key in ["gender", "commons", "image", "depictions"]:
        counter = counters[key]
        print(key)
        print(tabulate([counter], headers="keys", tablefmt="latex"), "\n\n")


def main(subset):
    path = DATA_ROOT_PATH / f"meerqat_{subset}" / "entities.json"
    with open(path) as file:
        entities = json.load(file)

    counters = count_entities(entities)
    output = DATA_ROOT_PATH / "visualization"
    output.mkdir(exist_ok=True)
    visualize_entities(counters, output, subset)


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    subset = args['<subset>']

    main(subset)