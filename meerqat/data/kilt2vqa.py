# coding: utf-8
"""Usage:
kilt2vqa.py ner <subset>
kilt2vqa.py ned <subset>
kilt2vqa.py count_entities <subset> [--threshold=<threshold>]
"""

from collections import Counter
import json
import numpy as np
import re
import spacy
from spacy.gold import align
from spacy.symbols import DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL
from spacy.symbols import dobj, nsubj, pobj, obj, nsubjpass, poss, obl, root

from docopt import docopt
from tqdm import tqdm
from tabulate import tabulate

from datasets import load_dataset, load_from_disk
from meerqat.data.loading import map_kilt_triviaqa, DATA_ROOT_PATH
from meerqat.visualization.utils import viz_spacy

INVALID_ENTITIES = {DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL}
VALID_DEP = {dobj, nsubj, pobj, obj, nsubjpass, poss, obl, root}
np.random.seed(0)


def wer(a, b):
    """Compute Word Error Rate (word-level Levenshtein distance) using spacy"""
    length = max(len(a), len(b))
    return align(a, b)[0] / length


def item2placeholder(item, model=None):
    """Make input question suitable for VQA
    by replacing an explicit entity mention and its syntactic children by a placeholder.
    e.g. 'Who wrote the opera Carmen?' -> 'Who wrote {mention}'
    Note that, not only the entity mention ('Carmen') but its syntactic children ('the opera')
    are replaced by the placeholder.

    The final goal is to find an image that represents the entity
    and fill the placeholder with an appropriate (ambiguous) mention (e.g. 'this opera', 'it')

    Parameters
    ----------
    item: dict
        original question should be in 'input' key
    model: spacy.lang.en.English
        Full spacy pipeline, we use both NER and dependency parsing

    Returns
    -------
    item: dict
        same as input with extra keys:
        - "placeholder": List[dict]
          One dict like {"input": str, "entity": dict, "dependency": str}
        - "spacy_input": dict
          Original input, POS and NER-tagged with spacy in dict format
          (using Doc.to_json())

    Usage
    -----
    hugging_face_dataset.map(item2placeholder, fn_kwargs={"model": spacy_model})
    """
    item['placeholder'] = []
    question = model(item['input'])
    item['spacy_input'] = question.to_json()
    # filter questions without entities
    if not question.ents:
        return item
    potential_questions = {}
    for e in question.ents:
        # filter invalid entities
        if e.label in INVALID_ENTITIES:
            continue
        for token in e:
            # filter invalid dependencies
            if token.dep not in VALID_DEP:
                continue
            # get leftmost and rightmost syntactic children
            # min/max hack is in case the "valid dependency token" is not the head in the entity span
            # e.g. "Who wrote the poem ‘The Lady of the Lake’?", "Lake" is pobj but a leaf
            start = min(token.left_edge.i, e.start)
            end = max(token.right_edge.i, e.end-1)
            potential_questions[(start, end)] = (e, token)
    # keep only the biggest span for overlapping mentions
    for (start, end), (e, token) in potential_questions.items():
        included = False
        for other_start, other_end in potential_questions:
            # included from the left
            if start >= other_start and end < other_end:
                included = True
            # included from the right
            elif start > other_start and end <= other_end:
                included = True
        if not included:
            # replace entity and its syntactic children by a placeholder
            placeholder = question[:start].text_with_ws + "{mention}" + token.right_edge.whitespace_ + question[end + 1:].text
            item['placeholder'].append({'input': placeholder,
                                        'entity': e.as_doc().to_json(),
                                        'dependency': token.dep_})
    return item


def stats(kilt_subset):
    stat_dict = {
        "placeholders": 0,
        "originals": len(kilt_subset),
        "distinct source": 0
    }
    for item in kilt_subset:
        len_placeholder = len(item["placeholder"])
        stat_dict["placeholders"] += len_placeholder
        stat_dict["distinct source"] += min(1, len_placeholder)

    return tabulate([stat_dict], headers="keys")


def stringify(kilt_subset, field="placeholder", include_answer=True, include_provenance=True, include_dep=False):
    results = []
    invalid = []
    for item in kilt_subset:
        if item[field]:
            result = [f"Q: {item['input']}"]
            for vq in item[field]:
                result.append(f"-> {vq['input']} {vq['dependency'] if include_dep else ''}")
            if include_answer:
                result.append(f"A: {item['output']['answer'][0]}")
            if include_provenance:
                result.append(f"\t{item['output']['provenance'][0]['title'][0]}")
            results.append("\n".join(result))
        else:
            invalid.append(viz_spacy(item['spacy_input']))
    return "\n\n\n".join(results), "\n".join(invalid)


def ner(subset):
    """
    1st step: Named Entity Recognition (NER):
    Goes through the kilt subset and apply 'item2placeholder' function (see its docstring)
    Save the resulting dataset to f"{DATA_ROOT_PATH}/meerqat_{subset}"
    """

    # load model and data
    model = spacy.load("en_core_web_lg")
    kilt_tasks = map_kilt_triviaqa()
    kilt_subset = kilt_tasks[subset]

    # go through the dataset and make input question suitable for VQA
    fn_kwargs = {"model": model}
    kilt_subset = kilt_subset.map(item2placeholder, fn_kwargs=fn_kwargs)
    print(stats(kilt_subset))

    # save data
    output_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    kilt_subset.save_to_disk(output_path)
    print(f"Successfully saved output to '{output_path}'")

    # show N random examples
    N = 100
    indices = np.arange(kilt_subset.shape[0])
    np.random.shuffle(indices)
    randoms = [kilt_subset[i.item()] for i in indices[:N]]
    results, invalid = stringify(randoms)
    print(f"\nGenerated questions out of {N} random examples:\n")
    print(results)
    print(f"\nPruned questions out of {N} random examples:\n")
    print(invalid)


def disambiguate(item, wikipedia, wikipedia_ids, pedia_index):
    """Go through candidate pages from TriviaQA and compute WER between entity mention and Wikipedia title/aliases
    One should filter entities with a minimal WER of 0.5 (see 'wer' key)
    """
    for vq in item["placeholder"]:
        ent = vq["entity"]['text'].lower().strip().split()
        wers = {}
        # process each wikipedia article only once (answer might come from different paragraphs but it's irrelevant for this)
        provenances = {provenance['wikipedia_id'][0]: re.sub("\(.+\)", "", provenance['title'][0].lower()).strip() for
                       provenance in item['output']['provenance']}
        for wid, title in provenances.items():
            aliases = {title}
            # get aliases from wikipedia
            pedia_index.setdefault(wid, np.where(wikipedia_ids == wid)[0].item())
            wiki_item = wikipedia[pedia_index[wid]]
            aliases.update({alias.lower().strip() for alias in wiki_item['wikidata_info']['aliases']['alias']})
            # compute WER and keep minimal for all aliases
            word_er = min([wer(ent, alias.split()) for alias in aliases])
            wers[wid] = word_er
        # keep minimal WER for all candidate articles
        best_provenance = min(wers, key=wers.get)
        best_wer = wers[best_provenance]
        wiki_item = wikipedia[pedia_index[best_provenance]]
        vq["entity"]['wikidata_info'] = wiki_item['wikidata_info']
        vq["entity"]['wikipedia_id'] = wiki_item['wikipedia_id']
        vq["entity"]["wer"] = best_wer
    return item


def ned(subset):
    """
    2nd step: Named Entity Disambiguation (NED) using TriviaQA provided list
    Assumes that you already ran NER and loads dataset from f"{DATA_ROOT_PATH}/meerqat_{subset}"
    and wikipedia from DATA_ROOT_PATH
    """
    # load data
    dataset = load_from_disk(DATA_ROOT_PATH / f"meerqat_{subset}")
    wikipedia = load_dataset('kilt_wikipedia', cache_dir=DATA_ROOT_PATH)['full']
    wikipedia_ids = np.array(wikipedia["wikipedia_id"])
    pedia_index = {}
    fn_kwargs = {"wikipedia": wikipedia, "wikipedia_ids": wikipedia_ids, "pedia_index": pedia_index}

    # go through dataset
    dataset = dataset.map(disambiguate, fn_kwargs=fn_kwargs)

    # save data
    output_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    dataset.save_to_disk(output_path)
    print(f"Successfully saved output to '{output_path}'")


def count_entities(subset, wer_threshold=0.5):
    path = DATA_ROOT_PATH / f"meerqat_{subset}"
    dataset = load_from_disk(path)
    entities = {}

    for item in tqdm(dataset):
        for vq in item['placeholder']:
            entity = vq['entity']
            if entity['wer'] > wer_threshold:
                continue
            wikidata_id = entity['wikidata_info']['wikidata_id']
            entities.setdefault(wikidata_id, {})
            entities[wikidata_id]["wikipedia_id"] = entity["wikipedia_id"]
            entities[wikidata_id].setdefault("n_questions", 0)
            entities[wikidata_id]["n_questions"] += 1

    output_path = path / "entities.json"
    with open(output_path) as file:
        json.dump(entities, file)
    print(f"\nSuccessfully saved output to {output_path}")
    values = np.array([entity["n_questions"] for entity in entities.values()])

    value_stats = [values.sum(), np.mean(values)]
    value_stats += np.quantile(values, [0., 0.25, 0.5, 0.75, 1.0])
    headers = ["sum", "mean", "min", "1st quartile", "median", "3rd quartile", "max"]
    print(tabulate([value_stats], headers=headers))


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    subset = args['<subset>']

    if args['ner']:
        ner(subset)
    elif args['ned']:
        ned(subset)
    elif args['count_entities']:
        wer_threshold = float(args['--threshold']) if args['--threshold'] else 0.5
        count_entities(subset, wer_threshold=wer_threshold)

