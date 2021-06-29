# MEERQAT
Source code and data used in my PhD/MEERQAT project.

# `data`

All the data should be stored there, although it will probably not be hosted on github (depending on the dataset size)

## Annotation of the data

Please refer to [`ANNOTATION.md`](./ANNOTATION.md) for the annotation instructions

# `meerqat`
This should contain all the source code and act as a python package (e.g. `import meerqat`)

## Installation

Install PyTorch following [the official document wrt to your distribution](https://pytorch.org/get-started/locally/) (preferably in a virtual environment)

```sh
git clone https://github.com/PaulLerner/meerqat.git
pip install -e meerqat
```


## `meerqat.data`

This should contain scripts to load the data, annotate it...

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

### `wiki.py`

Gathers data about entities mentioned in questions via Wikidata, Wikimedia Commons SPARQL services and Wikimedia REST API.

You should run all of these in this order to get the whole cake:

#### `wiki.py data entities <subset>` 
**input/output**: `entities.json` (output of `kilt2vqa.py count_entities`)  
queries many different attributes for all entities in the questions 

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

## `meerqat.visualization`

This should allow to visualize the data
