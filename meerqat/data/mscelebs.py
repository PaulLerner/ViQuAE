# coding: utf-8
"""Usage:
mscelebs.py entities <subset>
"""
import csv
import json

from pathlib import Path
from tqdm import tqdm
from docopt import docopt

from meerqat.data.loading import DATA_ROOT_PATH


def wikidata2freebase(mid):
    """For some reason the mid format on Wikidata is '/m/0xxx' but on Freebase it's 'm.0xxx'"""
    return "m." + mid[3:]


def freebase2wikidata(entities):
    """Map entities in a {mid: qid} dict"""
    mapping = {}
    for qid, entity in entities.items():
        mid = entity.get("freebase")
        if not qid:
            continue
        mid = wikidata2freebase(mid)
        mapping[mid] = qid
    return mapping


def count_entities(entities, mscelebs_path):
    fb2wd = freebase2wikidata(entities)
    with open(mscelebs_path) as file:
        reader = csv.reader(file, delimiter='\t')
        for row in tqdm(reader, desc="Reading MS-Celebs"):
            mid = row[0]
            qid = fb2wd.get(mid)
            if qid:
                entities[qid].setdefault("mscelebs", 0)
                entities[qid]["mscelebs"] += 1
    return entities


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    subset = args['<subset>']
    # load data
    subset_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    entities_path = subset_path / "entities.json"
    mscelebs_path = Path(DATA_ROOT_PATH, "MS-Celeb-1M/data/croped_face_images/FaceImageCroppedWithOutAlignment.tsv")
    with open(entities_path) as file:
        entities = json.load(file)

    if args['entities']:
        entities = count_entities(entities, mscelebs_path)

    with open(entities_path, "w") as file:
        json.dump(entities, file)
