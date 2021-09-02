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
import warnings

from docopt import docopt
from tqdm import tqdm
from collections import Counter
from pathlib import Path
import shutil
from tabulate import tabulate
import requests

from meerqat.data.loading import COMMONS_PATH
from meerqat.data.wiki import save_image


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
    session = requests.Session()
    for completions_path in completions_paths:
        for completion in load_completions(completions_path):
            vqa = retrieve_vqa(completion)
            discard = vqa.pop("discard", None)
            if discard is not None:
                counter[discard] += 1
                continue
            counter['ok'] += 1
            save_image(vqa['image'], session)
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
    disagreements = annotator_agreement(dataset)
    with open(output_path, 'w') as file:
        json.dump(dataset, file)
    with open(output_path.parent/f"{output_path.stem}-disagreements.json", 'w') as file:
        json.dump(disagreements, file)


def annotator_agreement(dataset):
    """

    Parameters
    ----------
    dataset : dict[str, list]
        output of merge

    Returns
    -------
    disagreements : dict
        questions of dataset where the annotators disagree on whether discarding the question or not
    """
    disagreements = {}
    counter = Counter()
    # store binary discard predictions
    # n[i] stores the number of annotators who discard the question or not
    n, P = [], []
    do_kappa = True
    num_annotators_per_annotation = None
    for meerqat_id, vqas in dataset.items():
        counter['total'] += 1
        if len(vqas) <= 1:
            continue
        if num_annotators_per_annotation is not None and len(vqas) != num_annotators_per_annotation:
            warnings.warn(f"Expected max. number of annotations to be {num_annotators_per_annotation}, got {len(vqas)} -> cannot compute Fleiss' Kappa")
            do_kappa = False
        num_annotators_per_annotation = len(vqas)
        counter['multiple_annotators'] += 1
        categories = dict(binary_discard=Counter(), binary_change_question=Counter(), binary_change_image=Counter())
        for vqa in vqas:
            discard = vqa.get("discard")
            if discard is not None:
                categories['binary_discard'][True] += 1
                # don't take into account question and image if the VQA triple was discarded
                continue
            categories['binary_discard'][False] += 1
            categories['binary_change_question'][vqa.get('vq', '').lower() != vqa['old_vq'].lower()] += 1
            categories['binary_change_image'][vqa['image'] != vqa['old_image']] += 1
        n_i = categories['binary_discard']
        n.append(n_i)
        total_pairs = num_annotators_per_annotation*(num_annotators_per_annotation-1)
        P.append((sum(n_ij**2 for n_ij in n_i.values())-num_annotators_per_annotation)/total_pairs)
        for category, values in categories.items():
            all_agree = len(values) == 1
            counter[category] += int(all_agree)
            if not all_agree and category == 'binary_discard':
                disagreements[meerqat_id] = {'vqas': vqas, 'annotator_agreement': categories}

    print(f"found {counter['multiple_annotators']} questions with at least 2 annotators over {counter['total']} questions")
    print(tabulate([counter], headers='keys'))
    if not do_kappa:
        return disagreements

    # compute Fleiss' Kappa computation
    # number of annotations with multiple annotators
    N = counter['multiple_annotators']
    # proportion of all discards and keeps
    p = sum(n, Counter())
    for k, v in p.items():
        p[k] /= (N*num_annotators_per_annotation)
    print(f'Proportion of discards (should sum to 1): {p[True]}, and keeps: {p[False]}')
    P_bar = sum(P)/N
    print(f'Average of P : {P_bar}')
    P_bar_e = sum(p_j**2 for p_j in p.values())
    print(f'Sum of squares of p: {P_bar_e}')
    kappa = (P_bar - P_bar_e)/(1 - P_bar_e)
    print(f"Fleiss' Kappa: {kappa}")
    return disagreements


def retrieve_vqa(completion):
    # "completions" was renamed to "annotations" in labelstudio 1.0
    completions = completion.get("completions", completion.get("annotations"))
    results = completions[0]["result"]
    data = completion["data"]
    vqa = dict(question=data["question"], wikidata_id=data["wikidata_id"], answer=data['answer'],
               image=data['image'], meerqat_id=data['meerqat_id'], old_vq=data['vq'], old_image=data['image'])

    # make a proper dict out of the results
    # note that "vq" is present in results even if the user didn't modify it (only absent if skipped)
    for result in results:
        key = result["from_name"]
        values = next(iter(result["value"].values()))
        # HACK: fix the bug where label-studio saves the original question when updating the annotation
        # see also ANNOTATION.md
        if len(values) != 1 and key == 'vq':
            values = set(values)
            if len(values) != 1:
                values.discard(data['vq'])
            value = next(iter(values))
            if len(values) != 1:
                warnings.warn(f"Found several values for '{key}': {values}")
        else:
            value = values[0]
        vqa[key] = value

    # update image if necessary
    change_image = vqa.pop("change_image", None)
    if change_image is not None:
        # e.g. "$altimage1caption" -> "altimage1"
        vqa['image'] = data[change_image[1: -7]]

    # HACK: fix users were not supposed to cancel task (skip button) without selecting a discard reason
    # see also ANNOTATION.md
    if completions[0].get('was_cancelled', False) and vqa.get('discard') is None:
        vqa['discard'] = 'cancelled'

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
