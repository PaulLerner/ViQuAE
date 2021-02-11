# coding: utf-8
"""Usage:
kilt2vqa.py <subset> [--n=<n> --prominent --abstract=<superclass_level> <classes_to_exclude>...]

Options:
--n=<n>                          Number of examples to visualize [default: 50].
--prominent                      Whether to consider only prominent depictions.
--abstract=<superclass_level>    Whether to filter out abstract entities based on the level of superclasses
<classes_to_exclude>...          Additional classes to exclude (e.g. "Q5 Q82794")
"""

from datasets import load_from_disk
import numpy as np
import json
from docopt import docopt
from tqdm import tqdm

from meerqat.data.loading import DATA_ROOT_PATH
from meerqat.data.wiki import keep_prominent_depictions, exclude_classes, QID_URI_PREFIX
from meerqat.data.kilt2vqa import generate_vqa

# HTML document format
HTML_FORMAT = """
<html>
<head>
    <link rel="stylesheet" href="styles.css">
</head>
<table>
    <tr>
        <th>Original Question</th>
        <th>Image</th>
        <th>Generated Question</th>
        <th>Answer</th>
    </tr>
    {tds}
</table>
</html>
"""
TD_FORMAT = """
<tr>
    <td>{original_question}</td>
    <td><img src="{url}" width="400"></td>
    <td>{generated_question}</td>
    <td>{answer}</td>
</tr>
"""


def write_html(dataset, visualization_path):
    tds = []
    for item in dataset:
        for vq in item['vq']:
            td = TD_FORMAT.format(
                original_question=item['input'],
                url=vq['url'],
                generated_question=vq['input'],
                answer=item['output']['answer'][0]
            )
            tds.append(td)
    html = HTML_FORMAT.format(tds="\n".join(tds))
    with open(visualization_path/'kilt2vqa.html', 'w') as file:
        file.write(html)


def subset2vqa(dataset, entities, n=50):
    indices = np.arange(dataset.shape[0])
    np.random.shuffle(indices)
    subset = []
    for i in tqdm(indices, desc="Generating visual questions"):
        item = generate_vqa(dataset[i.item()], entities)
        if not item['vq']:
            continue
        subset.append(item)
        if len(subset) >= n:
            break
    return subset


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    subset = args['<subset>']
    n = int(args['--n'])
    prominent_only = args['--prominent']
    superclass_level = int(args['--abstract']) if args['--abstract'] else None
    classes_to_exclude = set(QID_URI_PREFIX + qid for qid in args['<classes_to_exclude>'])
    visualization_path = DATA_ROOT_PATH / "visualization" / subset
    visualization_path.mkdir(exist_ok=True, parents=True)

    # load data
    subset_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    dataset = load_from_disk(subset_path)
    with open(subset_path/"entities.json") as file:
        entities = json.load(file)
    if prominent_only:
        entities = keep_prominent_depictions(entities)
    if superclass_level:
        with open(subset_path/f"{superclass_level}_superclasses.json") as file:
            superclasses = json.load(file)
        with open(DATA_ROOT_PATH/"abstract_entities.csv") as file:
            abstract_entities = set(line.split(",")[0] for line in file.read().split("\n")[1:] if line != '')
        classes_to_exclude.update(abstract_entities)
    entities = exclude_classes(entities, classes_to_exclude, superclasses)


    # generate a subset of VQA triples
    subset = subset2vqa(dataset, entities, n=n)

    # write result to HTML for visualization
    write_html(subset, visualization_path)

