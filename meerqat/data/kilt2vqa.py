# coding: utf-8
"""
========
Overview
========
.. image:: ../source_docs/kilt2vqa_big_picture.png

All the data should be stored in the `data` folder, at the root of this repo.

The goal is to generate questions suitable for VQA by replacing explicit entity mentions in existing textual QA datasets
by an ambiguous one and illustrate the question with an image (that depicts the entity).


-------
``ner``
-------
.. image:: ../source_docs/kilt2vqa_nlp.png

Slight misnomer, does a bit more than NER, i.e. dependency parsing.  
Detected entities with valid type and dependency are replaced by a placeholder along with its syntactic children.  
e.g. 'Who wrote *the opera **Carmen***?' &rarr; 'Who wrote `{mention}`'  
Note that, not only the entity mention ('Carmen') but its syntactic children ('the opera')
are replaced by the placeholder.
    

-------
``ner``
-------
.. image:: ../source_docs/kilt2vqa_nlp.png

Disambiguate entity mentions using Wikipedia pages provided in KILT.  
TriviaQA was originally framed as a reading-comprehension problem so the authors applied off-the-shelf NED and filtered
out pages that didn't contain the answer.  
For every entity mention we compute Word Error Rate (WER, i.e. word-level Levenshtein distance) for every wikipedia title
and aliases. We save the minimal match and WER and recommand filtering out WER > 0.5  
More data about these entities is gathered in `wiki.py`, 
just run `kilt2vqa.py count_entities` first to save a dict with all disambiguated entities (outputs `entities.json`).

---------------------
``generate mentions``
---------------------
.. image:: ../source_docs/kilt2vqa_mentiong_gen.png

Generate ambiguous entity mentions that can be used to replace the placeholder 
in the input question (you need to run `wiki.py data` first):  
    - if the gender is available (not animal sex):
        - 'this man' or 'this woman' (respecting transgender)
        - 'he/him/his' or 'she/her/hers' w.r.t mention dependency              
    - if human and occupation is available : 'this `{occupation}`' (respecting gender if relevant, e.g. for 'actress')
    - else if non-human:
        - if a taxon : 'this `{taxon rank}`' (e.g. 'species') 
        - else 'this `{class}`' (e.g. 'this tower')   
    
---------------
``generate vq``
---------------  
Make the VQA triple by choosing:  
    - uniformly a mention type and a mention from this mention type (generated in the previous step)  
    - the image with the best score (according to the heuristics computed in `wiki.py commons heuristics`).
      Tries to use a unique image per entity.

---------------
``labelstudio``
---------------
First calls `generate vq` i.e. no need to call both!  
The dataset is then converted to the Label Studio JSON format so you can annotate and convert the errors of the automatic pipeline (see [`ANNOTATION.md`](./ANNOTATION.md)).

------------
``download``
------------
Downloads images (set in `meerqat.data.wiki data entities`) from Wikimedia Commons using `meerqat.data.wiki.save_image`.  
This might take a while (thus the sharding options), any help/advice is appreciated :)
    
==============
For ``docopt``
==============
Usage:
kilt2vqa.py ner <subset> [--disable_caching]
kilt2vqa.py ned <subset> [--map_kwargs=<path> --disable_caching]
kilt2vqa.py generate mentions <subset> [--threshold=<threshold> --disable_caching]
kilt2vqa.py generate vq <subset> [--image_width=<n> --map_kwargs=<path> --disable_caching <categories_to_exclude>...]
kilt2vqa.py count_entities <subset> [--threshold=<threshold> --map_kwargs=<path> --disable_caching]
kilt2vqa.py labelstudio <subset> [--image_width=<n> --alternative_images=<n> --disable_caching <categories_to_exclude>...]
kilt2vqa.py download <subset> [--image_width=<n> --map_kwargs=<path> --disable_caching --num_shards=<n> --shard_index=<n>]

Options:
--threshold=<threshold>         Threshold for Word Error Rate (WER, i.e. word-level Levenshtein distance)
                                to consider the entity disambiguated [default: 0.5].
--alternative_images=<n>        Number of alternative images to provide [default: 8].
--image_width=<n>               Desired thumbnail width in pixels for the image url. Defaults to full-size
--map_kwargs=<path>             Path towards a JSON file containing key-words arguments for the dataset.map function (e.g. batch size)
--disable_caching               Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
--num_shards=<n>                Shard the dataset in n parts when downloading images
--shard_index=<n>               Index of the desired shard when downloading images (use along with --num_shards)

=========
Functions
=========
"""

import warnings
import json
import numpy as np
import pandas as pd
import random
import re
import spacy
try:
    from spacy.gold import align
except ImportError as e:
    warnings.warn(f"Got the following ImportError: {e}.\nTry using spacy==2.2.4")
from spacy.symbols import DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL
from spacy.symbols import dobj, nsubj, pobj, obj, nsubjpass, poss, obl, root

from docopt import docopt
from tqdm import tqdm
from tabulate import tabulate

import requests

from datasets import load_dataset, load_from_disk, set_caching_enabled
from .loading import map_kilt_triviaqa, DATA_ROOT_PATH
from .wiki import HUMAN, RESERVED_IMAGES, special_path_to_file_name, file_name_to_thumbnail, thumbnail_to_file_name, save_image
from .utils import md5

# spacy constants for NER
INVALID_ENTITIES = {DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL}
# TODO check root and obj: in practice, it never happened on TriviaQA dev set
VALID_DEP = {dobj, nsubj, pobj, obj, nsubjpass, poss, obl, root}

# spacy constants for pronoun-mention generation
HE_SHE_DEP = {spacy.symbols.NAMES[dep] for dep in [nsubj, nsubjpass]}
HIM_HER_DEP = {spacy.symbols.NAMES[dep] for dep in [dobj, obj, obl]}
HIS_HERS_DEP = {spacy.symbols.NAMES[poss]}

# Wikidata constants for pronoun-mention generation
#            'male'      'trans. male'
HE_GENDER = {'Q6581097', 'Q2449503'}
#             'female'    'trans. female'
SHE_GENDER = {'Q6581072', 'Q1052281'}
#            'intersex'  'non-binary'
NA_GENDER = {'Q1097630', 'Q48270'}
#             'male'    'female'
ANIMAL_SEX = {'Q44148', 'Q43445'}

# set random seed to get consistent random examples
np.random.seed(0)
random.seed(0)


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
        "distinct source": 0,
        "vqs": 0
    }
    for item in kilt_subset:
        len_placeholder = len(item["placeholder"])
        stat_dict["placeholders"] += len_placeholder
        stat_dict["distinct source"] += min(1, len_placeholder)
        stat_dict["vqs"] += len(item.get("vq", []))
        for vq in item['placeholder']:
            stat_dict.setdefault(vq['dependency'], 0)
            stat_dict[vq['dependency']] += 1

    return tabulate([stat_dict], headers="keys", tablefmt='latex')


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
            if include_provenance and item['output']['provenance']:
                result.append(f"\t{item['output']['provenance'][0]['title'][0]}")
            results.append("\n".join(result))
        else:
            invalid.append(item['spacy_input'])
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


def ned(subset, **map_kwargs):
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
    dataset = dataset.map(disambiguate, fn_kwargs=fn_kwargs, **map_kwargs)

    # save data
    output_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    dataset.save_to_disk(output_path)
    print(f"Successfully saved output to '{output_path}'")


def count_entities(subset, wer_threshold=0.5):
    path = DATA_ROOT_PATH / f"meerqat_{subset}"
    dataset = load_from_disk(path)
    entities = {}

    total, disambiguated = 0, 0
    for item in tqdm(dataset):
        for vq in item['placeholder']:
            total += 1
            entity = vq['entity']
            if entity['wer'] > wer_threshold:
                continue
            disambiguated += 1
            wikidata_id = entity['wikidata_info']['wikidata_id']
            entities.setdefault(wikidata_id, {})
            entities[wikidata_id]["wikipedia_id"] = entity["wikipedia_id"]
            entities[wikidata_id].setdefault("n_questions", 0)
            entities[wikidata_id]["n_questions"] += 1

    output_path = path / "entities.json"
    with open(output_path, 'w') as file:
        json.dump(entities, file)
    print(f"\nSuccessfully saved output to {output_path}")
    print(f"Disambiguated {disambiguated} questions ({len(entities)} unique entities) "
          f"out of {total} questions with a threshold of {wer_threshold}")
    print(pd.DataFrame([entity["n_questions"] for entity in entities.values()]).describe())


def generate_mention(item, entities, wer_threshold=0.5, feminine_labels={}):
    for vq in item["placeholder"]:
        entity = vq['entity']
        ambiguous_mentions = {
            "pronouns": [],
            "man_woman": [],
            "occupation": [],
            "instanceof": []
        }

        # filter ambiguous entities and skip filtered entities
        qid = entity['wikidata_info']['wikidata_id']
        entity_data = entities.get(qid)
        if entity['wer'] > wer_threshold or not entity_data:
            vq['ambiguous_mentions'] = ambiguous_mentions
            continue

        dependency = vq['dependency']

        gender = entity_data.get('gender', {}).get('value')
        gender = gender.split("/")[-1] if gender else gender
        human = HUMAN in entity_data.get('instanceof', {})
        taxon_rankLabel = entity_data.get('taxon_rankLabel', {}).get('value')
        # man_woman and pronouns
        if gender not in ANIMAL_SEX:
            # man_woman
            if gender in HE_GENDER:
                ambiguous_mentions["man_woman"].append("this man")
            elif gender in SHE_GENDER:
                ambiguous_mentions["man_woman"].append("this woman")
            elif gender in NA_GENDER or not gender:
                pass
            else:
                warnings.warn(f"No case were set for this gender: '{gender}'")

            # pronouns
            if dependency in HE_SHE_DEP:
                if gender in HE_GENDER:
                    ambiguous_mentions["pronouns"].append("he")
                elif gender in SHE_GENDER:
                    ambiguous_mentions["pronouns"].append("she")
            elif dependency in HIM_HER_DEP:
                if gender in HE_GENDER:
                    ambiguous_mentions["pronouns"].append("him")
                elif gender in SHE_GENDER:
                    ambiguous_mentions["pronouns"].append("her")
            elif dependency in HIS_HERS_DEP:
                if gender in HE_GENDER:
                    ambiguous_mentions["pronouns"].append("his")
                elif gender in SHE_GENDER:
                    ambiguous_mentions["pronouns"].append("hers")
            else:
                warnings.warn(f"No case were set for this dependency: '{dependency}'")

        # occupation
        if entity_data.get('occupation') and human:
            for occupation in entity_data['occupation'].values():
                feminine_label = feminine_labels.get(occupation['value'])
                if feminine_label and gender in SHE_GENDER:
                    occupation_label = feminine_label
                # default label is default value since most names in English don't have genders
                else:
                    occupation_label = occupation['label']['value']
                ambiguous_mentions['occupation'].append(f"this {occupation_label}")
        # taxon rank (e.g. "species") or class (aka instanceof)
        elif not human:
            # taxon rank
            if taxon_rankLabel:
                ambiguous_mentions['instanceof'].append(f"this {taxon_rankLabel}")
            # class (instanceof)
            else:
                for instanceof in entity_data.get('instanceof', {}).values():
                    feminine_label = feminine_labels.get(instanceof['value'])
                    if feminine_label and gender in SHE_GENDER:
                        instanceof_label = feminine_label
                    # default label is default value since most names in English don't have genders
                    else:
                        instanceof_label = instanceof['label']['value']
                    ambiguous_mentions['instanceof'].append(f"this {instanceof_label}")
        vq['ambiguous_mentions'] = ambiguous_mentions

    return item


def generate_mentions(subset, wer_threshold=0.5, **map_kwargs):
    """3rd step: generate ambiguous mentions given entities attributes (run `wiki.py data` first)"""
    # load data
    dataset_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    dataset = load_from_disk(dataset_path)
    with open(dataset_path / "entities.json", 'r') as file:
        entities = json.load(file)
    feminine_labels_path = dataset_path / "feminine_labels.json"
    if feminine_labels_path.exists():
        with open(feminine_labels_path, "r") as file:
            feminine_labels = json.load(file)
    else:
        feminine_labels = {}
    fn_kwargs = {
        "entities": entities,
        "wer_threshold": wer_threshold,
        "feminine_labels": feminine_labels
    }

    # go through dataset
    dataset = dataset.map(generate_mention, fn_kwargs=fn_kwargs, **map_kwargs)

    # save data
    dataset.save_to_disk(dataset_path)
    print(f"Successfully saved output to '{dataset_path}'")

    total, with_mention = 0, 0
    for item in dataset:
        for vq in item["placeholder"]:
            total += 1
            if [mention for mention_type in vq['ambiguous_mentions'] for mention in mention_type]:
                with_mention += 1
    print(f"{with_mention*100/total:.2f}% of the visual questions have at least one ambiguous mention")


def generate_vq(item, entities, image_width=512):
    """
    Generate a image (url), question, answer triple by choosing:
        - uniformly a mention type and a mention from this mention type
        - the image with the best score (with its title sorted last in "titles").
                Tries to use a unique image per entity.

    Parameters
    ----------
    item: Dataset item
    entities: dict (see wiki.py)
    image_width: int, optional
        desired thumbnail width in pixels for the image url
        Defaults to 512

    Returns
    -------
    item: Dataset item
        with a new 'vq' key (List[dict])
    """
    item['vq'] = []
    kilt_id = item['id']
    for placeholder in item['placeholder']:
        mention_types = [mention_type for mention_type in placeholder.get('ambiguous_mentions', {}).values() if mention_type]
        if not mention_types:
            continue
        qid = placeholder['entity']['wikidata_info']['wikidata_id']
        description = placeholder['entity']['wikidata_info']['description']
        # entity might have been filtered before-hand -> get qid instead of "[qid]"
        entity = entities.get(qid, {})
        titles = entity.get("titles")
        if not titles:
            continue
        # try to use unique images per entity -> pop titles
        if len(titles) > 1:
            # note we assume that the images are sorted in ascending order w.r.t. their score
            title = titles.pop()
        else:
            title = titles[0]
        url = file_name_to_thumbnail(title[len("File:"):], image_width=image_width)

        # choose mention type (e.g. pronoun or occupation) uniformly from all types (that are not empty)
        mention_type = random.choice(mention_types)
        # choose mention uniformly from all mentions in this type (e.g. Barack Obama is a politician and a statesperson)
        mention = random.choice(mention_type)

        inp = placeholder['input'].format(mention=mention)
        meerqat_id = md5("".join((kilt_id, qid, inp, url)))
        vq = {'input': inp,
              'url': url,
              'wikidata_id': qid,
              'meerqat_id': meerqat_id,
              'mentions': [mention for mention_type in mention_types for mention in mention_type],
              'description': description
              }
        item['vq'].append(vq)

    return item


def generate_vqs(subset, exclude_categories=set(), image_width=512, **map_kwargs):
    """
    Parameters
    ----------
    subset: str
        Name of the subset to load (e.g. validation_triviaqa)
    exclude_categories: set, optional
        Exclude image where these keywords are included in one of its categories
        e.g. {'cosplay'} might save you some trouble with GDPR
        Defaults to empty set (i.e. keep all)
    """
    # load data
    print("loading data...")
    dataset_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    dataset = load_from_disk(dataset_path)
    with open(dataset_path / "entities.json", 'r') as file:
        entities = json.load(file)

    # sort images and remove forbidden ones (move to wiki.py if it's too slow?)
    for entity in tqdm(entities.values(), desc="Processing images"):
        images = entity.get("images")
        if not images:
            continue

        # remove reserved images (e.g. illustrative_image) from the candidates
        for reserved_image_key in RESERVED_IMAGES:
            for reserved_image in map(special_path_to_file_name, entity.get(reserved_image_key, {})):
                images.pop(reserved_image, None)

        # Exclude image where these keywords are included in one of its categories
        if exclude_categories:
            todel = []
            for title, image in images.items():
                del_image = False
                image_categories = image.get("categories")
                if image_categories is None:
                    continue
                for image_category in image_categories:
                    image_category = image_category.lower()
                    for category_to_exclude in exclude_categories:
                        if category_to_exclude in image_category:
                            del_image = True
                            break
                    if del_image:
                        todel.append(title)
                        break
            for title in todel:
                images.pop(title)

        # sort images w.r.t. their score in ASCENDING order (allows simpler use of pop)
        entity["titles"] = sorted(images, key=lambda title: len(images[title]['heuristics']))

    # go through dataset
    dataset = dataset.map(generate_vq, fn_kwargs=dict(entities=entities, image_width=image_width), **map_kwargs)

    # save data
    dataset.save_to_disk(dataset_path)
    print(f"Successfully saved output to '{dataset_path}'")

    print(stats(dataset))

    return dataset, entities


def labelstudio(*args, image_width=512, alternative_images=8, **kwargs):
    """run generate_vqs and convert dataset to the Label Studio JSON format"""
    print("Generating visual questions...")
    dataset, entities = generate_vqs(*args, image_width=image_width, **kwargs)

    # convert dataset to the Label Studio JSON format
    ls = {}
    i = 0
    for item in tqdm(dataset, desc="Converting to Label Studio"):
        for vq in item["vq"]:
            # make some names more explicit and copy some stuff from original QA
            vq["image"] = vq.pop('url')
            title = thumbnail_to_file_name(vq["image"]).replace('_', ' ')
            caption = re.match(r"(.+)\.\w+", title)
            caption = caption.group(1) if caption is not None else title
            vq["image_caption"] = caption
            vq['question'] = item['input']
            vq["vq"] = vq.pop('input')
            vq['answer'] = item['output']['answer'][0]
            vq['mentions'] = ", ".join(vq['mentions'])
            qid = vq['wikidata_id']
            entity = entities[qid]
            vq['entityLabel'] = entity.get("entityLabel", {}).get("value", "")
            vq['entity_image'] = entity.get('reference_image', '')

            # gather alternative images to vq["image"]
            # remember images are sorted in ASC order wrt their score, thus the [::-1] to reverse the list
            for j, title in enumerate(entity["titles"][-alternative_images: ][::-1]):
                # remove "File:" prefix and extension
                caption = re.match(r"File:(.+)\.\w+", title)
                caption = caption.group(1) if caption is not None else title
                # title to url
                url = file_name_to_thumbnail(title[len("File:"):], image_width=image_width)
                vq[f"altimage{j}"] = url
                vq[f"altimage{j}caption"] = caption
            # no missing values: use empty string instead
            for j in range(j+1, alternative_images):
                vq[f"altimage{j}"] = ""
                vq[f"altimage{j}caption"] = ""

            ls[str(i)] = {"data": vq}
            i += 1

    # save output
    out_path = DATA_ROOT_PATH / f"meerqat_{subset}" / "labelstudio.json"
    with open(out_path, 'w') as file:
        json.dump(ls, file)
    print(f"Successfully saved output to '{out_path}'")


def download_image(item, session, image_width=512):
    file_name = thumbnail_to_file_name(item['url'])
    thumbnail = file_name_to_thumbnail(file_name, image_width=image_width)
    file_path = save_image(thumbnail, session)
    file_name = file_path.name if file_path is not None else None
    item['image'] = file_name
    return item


def download_images(subset, fn_kwargs, num_shards=None, shard_index=None, **map_kwargs):
    print("loading data...")
    dataset_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    dataset = load_from_disk(dataset_path)
    if num_shards is not None:
        dataset = dataset.shard(num_shards, shard_index)

    fn_kwargs.update(session=requests.Session())

    dataset = dataset.map(download_image, fn_kwargs=fn_kwargs, **map_kwargs)
    if num_shards is None:
        dataset.save_to_disk(dataset_path)
    else:
        dataset.save_to_disk(dataset_path/f"shard_{shard_index}_of_{num_shards}")


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    subset = args['<subset>']
    map_kwargs_path = args['--map_kwargs']
    if map_kwargs_path:
        with open(map_kwargs_path, 'r') as file:
            map_kwargs = json.load(file)
    else:
        map_kwargs = {}
    set_caching_enabled(not args['--disable_caching'])

    if args['ner']:
        ner(subset)
    elif args['ned']:
        ned(subset, **map_kwargs)
    elif args['count_entities']:
        wer_threshold = float(args['--threshold'])
        count_entities(subset, wer_threshold=wer_threshold)
    elif args['generate']:
        if args['mentions']:
            wer_threshold = float(args['--threshold'])
            generate_mentions(subset, wer_threshold=wer_threshold, **map_kwargs)
        elif args['vq']:
            exclude_categories = set(args['<categories_to_exclude>'])
            image_width = int(args['--image_width']) if args['--image_width'] is not None else None
            generate_vqs(subset, exclude_categories, image_width=image_width, **map_kwargs)
    elif args['labelstudio']:
        exclude_categories = set(args['<categories_to_exclude>'])
        alternative_images = int(args['--alternative_images'])
        image_width = int(args['--image_width']) if args['--image_width'] is not None else None
        labelstudio(subset, exclude_categories=exclude_categories, alternative_images=alternative_images, image_width=image_width)
    elif args['download']:
        image_width = int(args['--image_width']) if args['--image_width'] is not None else None
        num_shards = int(args['--num_shards']) if args['--num_shards'] is not None else None
        shard_index = int(args['--shard_index']) if args['--shard_index'] is not None else None
        download_images(subset, fn_kwargs=dict(image_width=image_width), num_shards=num_shards, shard_index=shard_index, **map_kwargs)

