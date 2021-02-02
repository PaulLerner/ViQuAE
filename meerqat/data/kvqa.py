# coding: utf-8
"""Usage:
kvqa.py <subset>
"""
import json

from pathlib import Path
from tqdm import tqdm
from docopt import docopt

from meerqat.data.loading import DATA_ROOT_PATH, KVQA_PATH


def update_entities(entities, kvqa):
    """Updates entities with data from KVQA, i.e. annotated images"""
    for question in tqdm(kvqa.values()):
        for qid in question['Qids']:
            entity = entities.get(qid)
            if not entity:
                continue
            entity.setdefault("kvqa", {})
            image = {
                # this path is relative to KVQA_PATH -> it should be used like KVQA_PATH/relative_path
                "relative_path": question['imgPath'],
                # is the entity (i.e. person) alone in the image ?
                "prominent": len(question['Qids'])==1
                # TODO add a position field to be appended to the mention if not (e.g. "on the left")
            }
            entity["kvqa"][question['imgPath']] = image
    return entities


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    subset = args['<subset>']

    subset_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    entities_path = subset_path / "entities.json"
    # load data
    with open(entities_path) as file:
        entities = json.load(file)
    with open(KVQA_PATH/"dataset.json") as file:
        kvqa = json.load(file)
    
    entities = update_entities(entities, kvqa)

    with open(entities_path, "w") as file:
        json.dump(entities, file)

