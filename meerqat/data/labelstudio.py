# coding: utf-8
"""Usage:
labelstudio.py save images [<path>...]
"""

import json
from docopt import docopt
from tqdm import tqdm
from collections import Counter
from pathlib import Path
from tabulate import tabulate

from meerqat.data.wiki import COMMONS_PATH, save_image


def save_images(completions_paths):
    COMMONS_PATH.mkdir(exist_ok=True)
    counter = Counter()
    progress = tqdm()
    for completions_path in completions_paths:
        completions_path = Path(completions_path)
        with open(completions_path, 'r') as file:
            completions = json.load(file)
        if not isinstance(completions, list):
            completions = [completions]
        for completion in completions:
            vqa = retrieve_vqa(completion)
            discard = vqa.pop("discard", None)
            if discard is not None:
                counter[discard] += 1
                continue
            counter['ok'] += 1
            save_image(vqa['image'])
            progress.update()
    progress.close()
    print(tabulate([counter], headers='keys'))


def retrieve_vqa(completion):
    # "completions" was renamed to "annotations" in labelstudio 1.0
    results = completion.get("completions", completion["annotations"])[0]["result"]
    data = completion["data"]
    vqa = dict(question=data["question"], wikidata_id=data["wikidata_id"], answer=data['answer'], image=data['image'])
    # make a proper dict out of the results
    # note that "vq" is always present in results, even if the user didn't modify it
    for result in results:
        key = result["from_name"]
        vqa[key] = next(iter(result["value"].values()))[0]

    # update image if necessary
    change_image = vqa.pop("change_image", None)
    if change_image is not None:
        # e.g. "$altimage1caption" -> "altimage1"
        vqa['image'] = data[change_image[1: -7]]

    return vqa


def main():
    args = docopt(__doc__)
    completions = args['<path>']
    if args['save']:
        if args['images']:
            save_images(completions)


if __name__ == '__main__':
    main()
