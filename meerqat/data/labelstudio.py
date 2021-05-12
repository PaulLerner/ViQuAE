# coding: utf-8
"""Usage:
labelstudio.py save images <path>...
labelstudio.py merge <output> <path>...
labelstudio.py assign <output> <todo> <start> <end> [--overlap=<n> --zip <config>...]

Options:
--overlap=<n>           Number of questions to leave in todo [default: 0].
--zip                   Whether to zip the output directory
"""

import json
from docopt import docopt
from tqdm import tqdm
from collections import Counter
from pathlib import Path
import shutil
from tabulate import tabulate

from meerqat.data.wiki import COMMONS_PATH, save_image


def load_completions(completions_path):
    """Handles individual completions path and JSON exports"""
    completions_path = Path(completions_path)
    with open(completions_path, 'r') as file:
        completions = json.load(file)
    if not isinstance(completions, list):
        completions = [completions]
    return completions


def save_images(completions_paths):
    COMMONS_PATH.mkdir(exist_ok=True)
    counter = Counter()
    progress = tqdm()
    for completions_path in completions_paths:
        for completion in load_completions(completions_path):
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


def merge(output_path, completions_paths):
    progress = tqdm()
    dataset = {}
    for completions_path in completions_paths:
        for completion in load_completions(completions_path):
            vqa = retrieve_vqa(completion)
            meerqat_id = vqa['meerqat_id']
            dataset.setdefault(meerqat_id, [])
            dataset[meerqat_id].append(vqa)
            progress.update()
    progress.close()
    annotator_agreement(dataset)
    with open(output_path, 'w') as file:
        json.dump(dataset, file)


def annotator_agreement(dataset):
    # TODO Fleiss' Kappa
    counter = Counter()
    for meerqat_id, vqas in dataset.items():
        counter['total'] += 1
        if len(vqas) <= 1:
            continue
        counter['multiple_annotators'] += 1
        categories = dict(binary_discard=Counter(), reason_discard=Counter(), binary_change_question=Counter(),
                          binary_change_image=Counter(), change_image=Counter())
        for vqa in vqas:
            discard = vqa.get("discard")
            if discard is not None:
                categories['binary_discard'][True] += 1
                categories['reason_discard'][discard] += 1
            else:
                categories['binary_discard'][False] += 1
            categories['binary_change_question'][vqa.get('vq', '').lower() != vqa['old_vq'].lower()] += 1
            if vqa['image'] != vqa['old_image']:
                categories['binary_change_image'][True] += 1
                categories['change_image'][vqa['image']] += 1
            else:
                categories['binary_change_image'][False] += 1
        for category, values in categories.items():
            # number of annotators who all agree on this question
            counter[category] += int(len(values) == 1)

    print(f"found {counter['multiple_annotators']} questions with at least 2 annotators over {counter['total']} questions")
    print(tabulate([counter], headers='keys'))


def retrieve_vqa(completion):
    # "completions" was renamed to "annotations" in labelstudio 1.0
    results = completion.get("completions", completion.get("annotations"))[0]["result"]
    data = completion["data"]
    vqa = dict(question=data["question"], wikidata_id=data["wikidata_id"], answer=data['answer'],
               image=data['image'], meerqat_id=data['meerqat_id'], old_vq=data['vq'], old_image=data['image'])
    # make a proper dict out of the results
    # note that "vq" is present in results even if the user didn't modify it (only absent if skipped)
    for result in results:
        key = result["from_name"]
        vqa[key] = next(iter(result["value"].values()))[0]

    # update image if necessary
    change_image = vqa.pop("change_image", None)
    if change_image is not None:
        # e.g. "$altimage1caption" -> "altimage1"
        vqa['image'] = data[change_image[1: -7]]

    return vqa


def assign_annotations(todo, start, end, overlap=0):
    assigned = {}
    for i in range(start, end - overlap):
        i = str(i)
        assigned[i] = todo.pop(i)
    for i in range(end - overlap, end):
        i = str(i)
        assigned[i] = todo[i]
    return assigned


def assign(output_path, todo_path, start, end, overlap=0, zip=False, configs=[]):
    with open(todo_path, 'r') as file:
        todo = json.load(file)

    assigned = assign_annotations(todo, start, end, overlap=overlap)

    output_path.mkdir(exist_ok=True, parents=True)
    with open(output_path/'tasks.json', 'w') as file:
        json.dump(assigned, file)
    for config in configs:
        shutil.copy(config, output_path)
    if zip:
        shutil.make_archive(output_path, 'zip', output_path)

    with open(todo_path, 'w') as file:
        json.dump(todo, file)


def main():
    args = docopt(__doc__)
    completions = args['<path>']

    if args['save']:
        if args['images']:
            save_images(completions)
    elif args['merge']:
        output_path = Path(args['<output>'])
        merge(output_path, completions)
    elif args['assign']:
        output_path, todo_path = Path(args['<output>']), Path(args['<todo>'])
        start, end, overlap = int(args['<start>']), int(args['<end>']), int(args['--overlap'])
        zip = args['--zip']
        configs = args['<config>']
        assign(output_path, todo_path, start, end, overlap=overlap, zip=zip, configs=configs)


if __name__ == '__main__':
    main()
