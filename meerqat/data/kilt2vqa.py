# coding: utf-8
"""Usage: kilt2vqa.py <subset>"""
import numpy as np
import spacy
from spacy.symbols import DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL
from spacy.symbols import dobj, nsubj, pobj, obj, nsubjpass, poss, obl, root

from docopt import docopt
from tabulate import tabulate

from meerqat.data.loading import map_kilt_triviaqa, DATA_ROOT_PATH
from meerqat.visualization.utils import viz_spacy

INVALID_ENTITIES = {DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL}
VALID_DEP = {dobj, nsubj, pobj, obj, nsubjpass, poss, obl, root}
np.random.seed(0)


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
            end = max(token.right_edge.i, e.end)
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


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    subset = args['<subset>']

    # load model and data
    model = spacy.load("en_core_web_lg")
    kilt_tasks = map_kilt_triviaqa()
    kilt_subset = kilt_tasks[subset]

    # go through the dataset and make input question suitable for VQA
    fn_kwargs = {"model": model}
    kilt_subset = kilt_subset.map(item2placeholder, fn_kwargs=fn_kwargs)
    print(stats(kilt_subset))

    # save data
    output_path = DATA_ROOT_PATH/f"meerqat_{subset}"
    kilt_subset.save_to_disk(output_path)
    print(f"Successfully saved output to '{output_path}'")

    # show N random examples
    N = 100
    indices = np.arange(kilt_tasks[subset].shape[0])
    np.random.shuffle(indices)
    randoms = [kilt_subset[i.item()] for i in indices[:N]]
    results, invalid = stringify(randoms)
    print(f"\nGenerated questions out of {N} random examples:\n")
    print(results)
    print(f"\nPruned questions out of {N} random examples:\n")
    print(invalid)
