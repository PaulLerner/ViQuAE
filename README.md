# MEERQAT
Source code and data used in my PhD/[MEERQAT project](https://www.meerqat.fr/).

In the paper [ViQuAE, a Dataset for Knowledge-based Visual Question Answering about Named Entities](https://openreview.net/forum?id=4YjpGcrcGy_), 
under **double-blind** review at ACL RR, the dataset has been renamed *ViQuAE*.
Until the paper is accepted, the name of the dataset will be *MEERQAT* here.

# `data`

All the data should be stored in the same folder, at the root of this repo.

The data is provided in two formats: HF's `datasets` (based on Apache Arrow) and plain-text JSONL files (one JSON object per line). 
Both formats can be used in the same way as `datasets` parses objects into python `dict` (see below), however our code only supports (and is heavily based upon) `datasets`.
Images are distributed separately, in standard formats (e.g. jpg).

Both dataset formats are distributed in two versions, with and without pre-computed features.
The pre-computed feature version allows you to skip one or several step described in [EXPERIMENTS.md](./EXPERIMENTS.md) (e.g. face detection).

Download the version you're interested in:

|           | with precomputed features                                               | without precomputed features                                            |  
|---------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------- |
|`datasets` | https://cea.talkspirit.com/#/l/permalink/drive/619d08e1eebc593c1716c623 | https://cea.talkspirit.com/#/l/permalink/drive/619d08e18989336a26769f8d |
|`JSONL`    | https://cea.talkspirit.com/#/l/permalink/drive/619d08e147c0e207220663af | https://cea.talkspirit.com/#/l/permalink/drive/619d08e1ded5af4ac84368ac |

Download the images from https://cea.talkspirit.com/#/l/permalink/drive/619d08a10a52e83dbe5ace7c
```sh
mkdir data
tar -xvzf /path/to/dataset.tar.gz --directory data
tar -xvzf /path/to/meerqat_dataset_Commons.tar.gz --directory data
# some part of the code expect that all of the images are stored in `data/Commons`
mv data/meerqat_dataset_Commons data/Commons
```

Instructions for the Knowledge Base (KB), coming soon!

The dataset format largely follows [KILT](https://huggingface.co/datasets/kilt_tasks). 
Here I’ll describe the dataset without pre-computed features. Pre-computed features are basically the output of each step described in [EXPERIMENTS.md](./EXPERIMENTS.md).

```py
In [1]: from datasets import load_from_disk
   ...: dataset = load_from_disk('meerqat_dataset_without_features')
In [2]: dataset
Out[2]: 
DatasetDict({
    train: Dataset({
        features: ['image', 'input', 'kilt_id', 'id', 'meta', 'original_question', 'output', 'url', 'wikidata_id'],
        num_rows: 1190
    })
    validation: Dataset({
        features: ['image', 'input', 'kilt_id', 'id', 'meta', 'original_question', 'output', 'url', 'wikidata_id'],
        num_rows: 1250
    })
    test: Dataset({
        features: ['image', 'input', 'kilt_id', 'id', 'meta', 'original_question', 'output', 'url', 'wikidata_id'],
        num_rows: 1257
    })
})
In [3]: item = dataset['test'][0]

# this is now a dict, like the JSON object loaded from the JSONL files
In [4]: type(item)
Out[4]: dict

# url of the grounding image
In [5]: item['url']
Out[5]: 'http://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Jackie_Wilson.png/512px-Jackie_Wilson.png'

# file name of the grounding image as stored in `data/Commons`
In [6]: item['image']
Out[6]: '512px-Jackie_Wilson.png'

# question string
In [7]: item['input']
Out[7]: "this singer's re-issued song became the UK Christmas number one after helping to advertise what brand?"

# answer string
In [8]: item['output']['original_answer']
Out[8]: "Levi's"

# processing the data:
In [9]: dataset.map(my_function)
# this is almost the same as (see how can you adapt the code if you don’t want to use the `datasets` library)
In [10]: for item in dataset:
    ...:     my_function(item)
```

## Annotation of the data

Please refer to [`ANNOTATION.md`](./ANNOTATION.md) for the annotation instructions

# Experiments

Please refer to [EXPERIMENTS.md](./EXPERIMENTS.md) for instructions to reproduce our experiments

# Reference

If you use this code or the MEERQAT dataset, please cite our paper:
```
TODO
```

# `meerqat`
This should contain all the source code and act as a python package (e.g. `import meerqat`)

## Installation

Install PyTorch 1.9.0 following [the official document wrt to your distribution](https://pytorch.org/get-started/locally/) (preferably in a virtual environment)

Also install [ElasticSearch](https://www.elastic.co/guide/en/elastic-stack-get-started/current/get-started-elastic-stack.html#install-elasticsearch) 
(and run it) if you want to do sparse retrieval.

The rest should be installed using `pip`:
```sh
git clone https://github.com/PaulLerner/meerqat.git
pip install -e meerqat
```

## `image`, `ir`, `models`, `train`
Those modules are best described along with the experiments, see [EXPERIMENTS.md](./EXPERIMENTS.md).

## `meerqat.data`

This should contain scripts to load the data, annotate it...

### `loading`
This is probably the only file in `data` interesting for the users of the dataset.

#### `passages` 
Segments Wikipedia articles (from the `kilt_wikipedia` dataset) into passages (e.g. paragraphs)
Current options (passed in a JSON file) are:
 - `prepend_title`: whether to prepend the title at the beginning of each passage like `"<title> [SEP] <passage>"`
 - `special_fields`: removes the title, sections titles ("Section::::") and bullet-points ("BULLET::::")
 - `uniform`: each passage is `n` tokens, without overlap. Tokenized with a `transformers` tokenizer
 - `uniform_sents`: each article is first segmented into sentences using `spacy`. 
                    Then sentences are grouped into passage s.t. each passage holds a maximum of `n` tokens 
                    (`spacy` tokens here, not `transformers` like above)

#### `map`
Make a JSON file out of a `dataset` column for quick (and string) indexing.
 
### `kilt2vqa.py`

The goal is to generate questions suitable for VQA by replacing explicit entity mentions in existing textual QA datasets
 by an ambiguous one and illustrate the question with an image (that depicts the entity).

[4 steps](./figures/kilt2vqa_big_picture.png) (click on the links to see the figures):
1. [`ner`](./figures/kilt2vqa_nlp.png) - Slight misnomer, does a bit more than NER, i.e. dependency parsing.  
    Detected entities with valid type and dependency are replaced by a placeholder along with its syntactic children.  
    e.g. 'Who wrote *the opera **Carmen***?' &rarr; 'Who wrote `{mention}`'  
    Note that, not only the entity mention ('Carmen') but its syntactic children ('the opera')
    are replaced by the placeholder.
2. [`ned`](./figures/kilt2vqa_nlp.png) - Disambiguate entity mentions using Wikipedia pages provided in KILT.  
    TriviaQA was originally framed as a reading-comprehension problem so the authors applied off-the-shelf NED and filtered
    out pages that didn't contain the answer.  
    For every entity mention we compute Word Error Rate (WER, i.e. word-level Levenshtein distance) for every wikipedia title
    and aliases. We save the minimal match and WER and recommand filtering out WER > 0.5  
    More data about these entities is gathered in `wiki.py`, 
    just run `kilt2vqa.py count_entities` first to save a dict with all disambiguated entities (outputs `entities.json`).
3. [`generate mentions`](./figures/kilt2vqa_mentiong_gen.png) - Generate ambiguous entity mentions that can be used to replace the placeholder in the input question 
    (you need to run `wiki.py data` first):  
    - if the gender is available (not animal sex):
        - 'this man' or 'this woman' (respecting transgender)
        - 'he/him/his' or 'she/her/hers' w.r.t mention dependency              
    - if human and occupation is available : 'this `{occupation}`' (respecting gender if relevant, e.g. for 'actress')
    - else if non-human:
        - if a taxon : 'this `{taxon rank}`' (e.g. 'species') 
        - else 'this `{class}`' (e.g. 'this tower')        
4.  `generate vq` - make the VQA triple by choosing:  
        - uniformly a mention type and a mention from this mention type (generated in the previous step)  
        - the image with the best score (according to the heuristics computed in `wiki.py commons heuristics`).
          Tries to use a unique image per entity.

`labelstudio` first calls `generate vq` i.e. no need to call both!  
The dataset is then converted to the Label Studio JSON format so you can annotate and convert the errors of the automatic pipeline (see [`ANNOTATION.md`](./ANNOTATION.md)).

`download` downloads images (set in `meerqat.data.wiki data entities`) from Wikimedia Commons using `meerqat.data.wiki.save_image`.  
This might take a while (thus the sharding options), any help/advice is appreciated :)

### `wiki.py`

Gathers data about entities mentioned in questions via Wikidata, Wikimedia Commons SPARQL services and Wikimedia REST API.

You should run all of these in this order to get the whole cake:

#### `wiki.py data entities <subset>` 
**input/output**: `entities.json` (output of `kilt2vqa.py count_entities`)  
queries many different attributes for all entities in the questions 

Also sets a 'reference image' to the entity using Wikidata properties in the following order of preference:
- P18 ‘image’ (it is roughly equivalent to the infobox image in Wikipedia articles)
- P154 ‘logo image’
- P41 ‘flag image’
- P94 ‘coat of arms image’
- P2425 ‘service ribbon image’

#### `wiki.py data feminine <subset>` 
**input**: `entities.json`  
**output**: `feminine_labels.json`  
gets feminine labels for classes and occupations of these entities

#### `wiki.py data superclasses <subset> [--n=<n>]` 
**input**: `entities.json`  
**output**: `<n>_superclasses.json`  
gets the superclasses of the entities classes up `n` level (defaults to 'all', i.e. up to the root)
#### (OPTIONAL) we found that heuristics/images based on depictions were not that discriminative
##### `wiki.py commons sparql depicts <subset>`
**input/output**: `entities.json`  
Find all images in Commons that *depict* the entities
##### `wiki.py commons sparql depicted <subset>`
**input**: `entities.json`  
**output**: `depictions.json`  
Find all entities depicted in the previously gathered step
##### `wiki.py data depicted <subset>` 
**input**: `entities.json`, `depictions.json`   
**output**: `entities.json`  
Gathers the same data as in `wiki.py data entities <subset>` for *all* entities depicted in any of the depictions  
Then apply a heuristic to tell whether an image depicts the entity prominently or not: 
> *the depiction is prominent if the entity is the only one of its class*  
  e.g. *pic of Barack Obama and Joe Biden* -> not prominent  
       *pic of Barack Obama and the Eiffel Tower* -> prominent  

Note this heuristic is not used in `commons heuristics`

#### `wiki.py filter <subset> [--superclass=<level> --positive --negative --deceased=<year> <classes_to_exclude>...]`
**input/output**: `entities.json`  
Filters entities w.r.t. to their class/nature/"instance of" and date of death, see `wiki.py` docstring for option usage (TODO share concrete_entities/abstract_entities)
Also entities with a ‘sex or gender’ (P21) or ‘occupation’ (P106) are kept by default.

Note this deletes data so maybe save it if you're unsure about the filter.

#### `wiki.py commons rest <subset> [--max_images=<max_images> --max_categories=<max_categories>]`
**input/output**: `entities.json`  

Gathers images and subcategories recursively from the entity root commons-category

Except if you have a very small dataset you should probably set `--max_images=0` to query only categories and use `wikidump.py` to gather images from those.  
`--max_categories` defaults to 100.

#### `wiki.py commons heuristics <subset> [<heuristic>...]`
**input/output**: `entities.json`  
Run `wikidump.py` first to gather images.  
Compute heuristics for the image (control with `<heuristic>`, default to all):
- `categories`: the entity label should be included in *all* of the image category
- `description`: the entity label should be included in the image description
- `title`: the entity label should be included in the image title/file name
- `depictions`: the image should be tagged as *depicting* the entity (gathered in `commons sparql depicts`)

### `wikidump.py`
**input/output**: `entities.json`  
Usage: `wikidump.py <subset>`  
Parses the dump (should be downloaded first, TODO add instructions), gathers images and assign them to the relevant entity given its common categories (retrieved in `wiki.py commons rest`)  
Note that the wikicode is parsed very lazily and might need a second run depending on your application, e.g. templates are not expanded...

### `labelstudio`
Used to manipulate the output of [Label Studio](https://labelstud.io/), see also [ANNOTATION.md](./ANNOTATION.md)  
- `assign` takes annotations out of the TODO list in a `tasks.json` file (input to LS)
- `save images` similar to `kilt2vqa download`, not used for the final dataset
- `merge` merges several LS outputs, also compute inter-annotator agreement and saves disagreements
- `agree` merges the output of `merge` along with the corrected disagreements


## `meerqat.visualization`

This should allow to visualize the data
