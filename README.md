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
    just run `kilt2vqa.py count_entities` first to save a dict with all disambiguated entities.
3. `generate mentions` - Generate ambiguous entity mentions that can be used to replace the placeholder in the input question 
    (you need to run `wiki.py data` first):  
    - if the gender is available (not animal sex):
        - 'this man' or 'this woman' (respecting transgender)
        - 'he/him/his' or 'she/her/hers' w.r.t mention dependency              
    - if human and occupation is available : 'this `{occupation}`'
    - else if non-human:
        - 'this `{instance of}`'        

### `wiki.py`

Gathers data about entities mentioned in questions via Wikidata and Wikimedia Commons SPARQL services.

## `meerqat.visualization`

This should allow to visualize the data