# MEERQAT
Source code and data used in my PhD/MEERQAT project.

# `data`

All the data should be stored there, although it will probably not be hosted on github (depending on the dataset size)

# `meerqat`
This should contain all the source code and act as a python package (e.g. `import meerqat`)

## `meerqat.data`

This should contain scripts to load the data, annotate it...

### `kilt2vqa.py`

The goal is to generate questions suitable for VQA by replacing explicit entity mentions in existing textual QA datasets
 by an ambiguous one and illustrate the question with an image (that depicts the entity).

3 steps :
1. `ner` - Slight misnomer, does a bit more than NER, i.e. dependency parsing.  
    Detected entities with valid type and dependency are replaced by a placeholder along with its syntactic children.  
    e.g. 'Who wrote *the opera **Carmen***?' &rarr; 'Who wrote `{mention}`'  
    Note that, not only the entity mention ('Carmen') but its syntactic children ('the opera')
    are replaced by the placeholder.
2. `ned` - Disambiguate entity mentions using Wikipedia pages provided in KILT.  
    TriviaQA was originally framed as a reading-comprehension problem so the authors applied off-the-shelf NED and filtered
    out pages that didn't contain the answer.  
    For every entity mention we compute Word Error Rate (WER, i.e. word-level Levenshtein distance) for every wikipedia title
    and aliases. We save the minimal match and WER and recommand filtering out WER > 0.5  
    More data about these entities is gathered in `wiki.py`, 
    just run `kilt2vqa.py count_entities` first to save a dict with all disambiguated entities (outputs `entities.json`).
3. `generate mentions` - Generate ambiguous entity mentions that can be used to replace the placeholder in the input question 
    (you need to run `wiki.py data` first):  
    - if the gender is available (not animal sex):
        - 'this man' or 'this woman' (respecting transgender)
        - 'he/him/his' or 'she/her/hers' w.r.t mention dependency              
    - if human and occupation is available : 'this `{occupation}`' (respecting gender if relevant, e.g. for 'actress')
    - else if non-human:
        - if a taxon : 'this `{taxon rank}`' (e.g. 'species') 
        - else 'this `{class}`' (e.g. 'this tower')        

### `wiki.py`

Gathers data about entities mentioned in questions via Wikidata and Wikimedia Commons SPARQL services 
(`wiki.py commons rest` that uses Wikimedia REST API is intractable atm).

You should run all of these in this order to get the whole cake:
#### `wiki.py data entities <subset>` 
**input/ouput**: `entities.json` (output of `kilt2vqa.py count_entities`)  
queries many different attributes for all entities in the questions 
#### `wiki.py data feminine <subset>` 
**input**: `entities.json`  
**ouput**: `feminine_labels.json`  
gets feminine labels for classes and occupations of these entities
#### `wiki.py data superclasses <subset> [--n=<n>]` 
**input**: `entities.json`  
**ouput**: `<n>_superclasses.json`  
gets the superclasses of the entities classes up `n` level (defaults to 'all', i.e. up to the root)
#### `wiki.py commons sparql depicts <subset>`
**input/ouput**: `entities.json`  
Find all images in Commons that *depict* the entities
#### `wiki.py commons sparql depicted <subset>`
**input**: `entities.json`  
**ouput**: `depictions.json`  
Find all entities depicted in the previously gathered step
#### `wiki.py data depicted <subset>` 
**input**: `entities.json`, `depictions.json`   
**ouput**: `entities.json`  
Gathers the same data as in `wiki.py data entities <subset>` for *all* entities depicted in any of the depictions  
Then apply a heuristic to tell whether an image depicts the entity prominently or not: 
> *the depiction is prominent if the entity is the only one of its class*  
  e.g. *pic of Barack Obama and Joe Biden* -> not prominent  
       *pic of Barack Obama and the Eiffel Tower* -> prominent  

## `meerqat.visualization`

This should allow to visualize the data