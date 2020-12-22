# coding: utf-8
"""Usage: kilt2vqa.py <subset>"""
import numpy as np
import spacy
from spacy.symbols import DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL
from spacy.symbols import dobj, nsubj, pobj, obj, nsubjpass, poss, obl, root

from docopt import docopt
from tabulate import tabulate

from meerqat.data.loading import map_kilt_triviaqa
from meerqat.visualization.utils import viz_spacy

INVALID_ENTITIES = {DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL}
VALID_DEP = {dobj, nsubj, pobj, obj, nsubjpass, poss, obl, root}
np.random.seed(0)


def subset2placeholder(kilt_subset, model):
    """Goes through a KILT subset (e.g. TriviaQA) and make the question suitable for VQA
    by replacing an explicit entity mention and its syntactic children by a placeholder.
    e.g. 'Who wrote the opera Carmen?' -> 'Who wrote {mention}'
    Note that, not only the entity mention ('Carmen') but its syntactic children ('the opera')
    are replaced by the placeholder.

    The final goal is to find an image that represents the entity
    and fill the placeholder with an appropriate (ambiguous) mention (e.g. 'this opera', 'it')

    Parameters
    ----------
    kilt_subset: List[dict]
        original question should be in 'input' key
    model: spacy.lang.en.English
        Full spacy pipeline, we use both NER and dependency parsing

    Returns
    -------
    kilt_subset: List[dict]
        same as input with extra keys:
        - "placeholder": List[dict]
          One dict like {"input": str, "entity": Span, "dependency": str}
        - "spacy_input": Doc

    Note
    ----
    kilt_subset should likely be a datasets.Dataset and not a List[dict]
    """
    for item in kilt_subset:
        item['placeholder'] = []
        item['spacy_input'] = model(item['input'])
        question = item['spacy_input']
        # filter questions without entities
        if not question.ents:
            continue
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
                start, end = token.left_edge.i, token.right_edge.i
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
                                            'entity': e,
                                            'dependency': token.dep_})
    return kilt_subset


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


if __name__ == '__main__':
    args = docopt(__doc__)
    subset = args['<subset>']
    model = spacy.load("en_core_web_lg")
    kilt_tasks = map_kilt_triviaqa()

    # test on a random subset
    # TODO save the data using pyarrow
    indices = np.arange(kilt_tasks[subset].shape[0])
    np.random.shuffle(indices)
    kilt_subset = [kilt_tasks[subset][i.item()] for i in indices[:100]]
    kilt_subset = subset2placeholder(kilt_subset, model)
    print(stats(kilt_subset))
    results, invalid = stringify(kilt_subset)
    print(results)
    print(invalid)
