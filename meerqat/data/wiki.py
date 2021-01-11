# coding: utf-8
"""Usage:
wiki.py data <subset>
"""
import json
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
from docopt import docopt

from meerqat.data.loading import DATA_ROOT_PATH

# Template for wikidata to query 'instance of' (P31), 'commons category' (P373),
# 'image' (P18), 'occupation' (P106) and 'gender' (P21) given a list of entities
# should be used like 'WIKIDATA_QUERY % "wd:Q76 wd:Q78579194 wd:Q42 wd:Q243"'
# i.e. entity ids are space-separated and prefixed by 'wd:'
WIKIDATA_QUERY = """
SELECT ?entity ?entityLabel ?instanceof ?instanceofLabel ?commons ?image ?occupation ?occupationLabel ?gender ?genderLabel
{
  VALUES ?entity { %s }
  OPTIONAL{ ?entity wdt:P373 ?commons . }
  ?entity wdt:P31 ?instanceof .
  OPTIONAL { ?entity wdt:P18 ?image . }
  OPTIONAL { ?entity wdt:P21 ?gender . }
  OPTIONAL { ?entity wdt:P106 ?occupation . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"


def data(wikidata_ids):
    """
    Query WIKIDATA_QUERY ('instance of', 'commons category', ...) for entities
    in wikidata_ids

    Parameters
    ----------
    wikidata_ids: List[str]
        List of wikidata QIDs
    """
    sparql = SPARQLWrapper(WIKIDATA_ENDPOINT)
    sparql.setReturnFormat(JSON)
    results, qids = [], []
    # query only 100 qid at a time
    for i, qid in enumerate(tqdm(wikidata_ids, desc="Querying Wikidata")):
        qids.append(f"wd:{qid}")
        if (i+1) % 100 == 0 or i == (len(wikidata_ids) - 1):
            sparql.setQuery(WIKIDATA_QUERY % " ".join(qids))
            results += sparql.query().convert()['results']['bindings']
            qids = []
    print(f"Query succeeded! Got {len(results)} results")

    return results


def update_from_data(subset):
    """
    Loads entities from {DATA_ROOT_PATH}/meerqat_{subset}/entities.json and updates them with info
    queried in 'data' (from wikidata)
    """
    # load entities
    path = DATA_ROOT_PATH / f"meerqat_{subset}" / "entities.json"
    with open(path) as file:
        entities = json.load(file)

    # query wikidata
    results = data(entities.keys())

    # update entities with results
    for result in tqdm(results, desc="Updating entities"):
        qid = result['entity']['value'].split('/')[-1]
        # handle keys/attributes that are unique
        for unique_key in ({'entityLabel', 'gender', 'genderLabel', 'image', 'commons'} & result.keys()):
            # simply add or update the key/attribute
            entities[qid][unique_key] = result[unique_key]
        # handle keys/attributes that may be multiple
        for multiple_key in ({'instanceof', 'occupation'} & result.keys()):
            # create a new dict for this key/attribute so we don't duplicate data
            entities[qid].setdefault(multiple_key, {})
            # store corresponding label in the 'label' field
            result[multiple_key]['label'] = result[multiple_key + 'Label']
            # value (e.g. QID) of the attribute serves as key
            multiple_value = result[multiple_key]['value']
            entities[qid][multiple_key][multiple_value] = result[multiple_key]

    with open(path, 'w') as file:
        json.dump(entities, file)

    print(f"Successfully saved output to {path}")

if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    subset = args['<subset>']

    if args['data']:
        update_from_data(subset)
