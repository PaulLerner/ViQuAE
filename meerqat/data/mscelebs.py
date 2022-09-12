# coding: utf-8
"""Usage:
mscelebs.py entities <subset>
mscelebs.py extract
"""
import csv
import json
import base64
import csv

from pathlib import Path
from tqdm import tqdm
from docopt import docopt

from .loading import DATA_ROOT_PATH


TOTAL_LINES = 8456240

def wikidata2freebase(mid):
    """For some reason the mid format on Wikidata is '/m/0xxx' but on Freebase it's 'm.0xxx'"""
    return "m." + mid[3:]


def freebase2wikidata(entities):
    """Map entities in a {mid: qid} dict"""
    mapping = {}
    for qid, entity in entities.items():
        mid = entity.get("freebase")
        if not mid:
            continue
        mid = wikidata2freebase(mid['value'])
        mapping[mid] = qid
    return mapping


def count_entities(entities, mscelebs_path):
    fb2wd = freebase2wikidata(entities)
    with open(mscelebs_path) as file:
        reader = csv.reader(file, delimiter='\t')
        for row in tqdm(reader, desc="Reading MS-Celebs", total=TOTAL_LINES):
            mid = row[0]
            qid = fb2wd.get(mid)
            if qid:
                entities[qid].setdefault("mscelebs", 0)
                entities[qid]["mscelebs"] += 1
    return entities


def extract(mscelebs_path):
    """Based on https://github.com/cmusatyalab/openface/blob/master/data/ms-celeb-1m/extract.py"""
    output_dir = mscelebs_path.parent/"jpgs"
    output_dir.mkdir(exist_ok=True)
    with open(mscelebs_path) as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        for row in tqdm(reader, desc="Extracting images", total=TOTAL_LINES):
            mid, imgSearchRank, faceID, data = row[0], row[1], row[4], base64.b64decode(row[-1])

            entity_dir = output_dir / mid
            entity_dir.mkdir(exist_ok=True)
            output_path = entity_dir / f"{imgSearchRank}-{faceID}.jpg"

            with open(output_path, 'wb') as jpg:
                jpg.write(data)


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    subset = args['<subset>']

    mscelebs_path = Path(DATA_ROOT_PATH, "MS-Celeb-1M/data/croped_face_images/FaceImageCroppedWithOutAlignment.tsv")
    subset_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    entities_path = subset_path / "entities.json"

    if args['extract']:
        extract(mscelebs_path)
    elif args['entities']:
        # load data
        with open(entities_path) as file:
            entities = json.load(file)
        entities = count_entities(entities, mscelebs_path)
        with open(entities_path, "w") as file:
            json.dump(entities, file)
