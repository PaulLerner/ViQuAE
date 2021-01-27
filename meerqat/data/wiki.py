# coding: utf-8
"""Usage:
wiki.py data <subset>
wiki.py commons sparql depicts <subset>
wiki.py commons sparql depicted <subset>
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

# template for beta-commons SPARQL API to query images that depict (P180) entities
# same usage as WIKIDATA_QUERY
COMMONS_SPARQL_QUERY = """
SELECT ?depicted_entity ?commons_entity ?special_path ?url ?encoding WHERE {
  VALUES ?depicted_entity { %s }
  ?commons_entity wdt:P180 ?depicted_entity .
  ?commons_entity schema:contentUrl ?url .
  ?commons_entity schema:encodingFormat ?encoding .
  # restrict media to be images handleable by PIL.Image
  VALUES ?encoding { "image/png" "image/jpg" "image/jpeg" "image/tiff" "image/gif" }
  bind(iri(concat("http://commons.wikimedia.org/wiki/Special:FilePath/", wikibase:decodeUri(substr(str(?url),53)))) AS ?special_path)
}
"""
# query entities depicted in images given image identifier (see above for more details)
COMMONS_DEPICTED_ENTITIES_QUERY = """
SELECT ?commons_entity ?depicted_entity WHERE {
  VALUES ?commons_entity { %s }
  ?commons_entity wdt:P180 ?depicted_entity .
}
"""
COMMONS_SPARQL_ENDPOINT = "https://wcqs-beta.wmflabs.org/sparql"


def query_sparql_entities(query, endpoint, wikidata_ids, prefix='wd:',
                          n=100, return_format=JSON, description=None):
    """
    Queries query%entities by batch of n (defaults 100)
    where entities is n QIDs in wikidata_ids space-separated and prefixed by prefix
    (should be 'wd:' for Wikidata entities and 'sdc:' for Commons entities)

    Returns query results
    """
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(return_format)
    results, qids = [], []
    # query only n qid at a time
    for i, qid in enumerate(tqdm(wikidata_ids, desc=description)):
        qids.append(prefix+qid)
        if (i + 1) % n == 0 or i == (len(wikidata_ids) - 1):
            sparql.setQuery(query % " ".join(qids))
            results += sparql.query().convert()['results']['bindings']
            qids = []
    print(f"Query succeeded! Got {len(results)} results")

    return results


def update_from_data(entities):
    """Updates entities with info queried in from Wikidata"""

    # query Wikidata
    results = query_sparql_entities(WIKIDATA_QUERY, WIKIDATA_ENDPOINT, entities.keys(),
                                    description="Querying Wikidata")

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

    return entities


def update_from_commons_sparql(entities):
    # query Wikimedia Commons
    results = query_sparql_entities(COMMONS_SPARQL_QUERY, COMMONS_SPARQL_ENDPOINT,
                                    entities.keys(),
                                    description="Querying Wikimedia Commons")

    # update entities with results
    for result in tqdm(results, desc="Updating entities"):
        qid = result['depicted_entity']['value'].split('/')[-1]
        commons_qid = result['commons_entity']['value']
        # create a new key 'depictions' to store depictions in a dict
        entities[qid].setdefault("depictions", {})
        # use commons_qid (e.g. https://commons.wikimedia.org/entity/M88412327) as key in this dict
        entities[qid]["depictions"].setdefault(commons_qid, {})
        entities[qid]["depictions"][commons_qid]['url'] = result['url']
        entities[qid]["depictions"][commons_qid]['special_path'] = result['special_path']

    return entities


def query_depicted_entities(depictions):
    # query Wikimedia Commons
    results = query_sparql_entities(COMMONS_SPARQL_QUERY, COMMONS_SPARQL_ENDPOINT,
                                    depictions.keys(), prefix="sdc:",
                                    description="Querying Wikimedia Commons")
    # update depictions with results
    for result in tqdm(results, desc="Updating depictions"):
        qid = result['commons_entity'].split('/')[-1]
        depictions[qid].add(result["depicted_entity"])
    return depictions


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    subset = args['<subset>']

    # load entities
    subset_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    path = subset_path / "entities.json"
    with open(path) as file:
        entities = json.load(file)

    # update from Wikidata or Wikimedia Commons
    if args['data']:
        entities = update_from_data(entities)
        # save output
        with open(path, 'w') as file:
            json.dump(entities, file)
    elif args['commons']:
        if args['sparql']:
            # find images that depict the entities
            if args['depicts']:
                entities = update_from_commons_sparql(entities)
                # save output
                with open(path, 'w') as file:
                    json.dump(entities, file)
            # find entities depicted in the images
            elif args['depicted']:
                # get depictions
                depictions = {depiction.split('/')[-1]: set()
                              for entity in entities.values()
                              for depiction in entity.get("depictions", {})}
                depictions = query_depicted_entities(depictions)
                # save output
                path = subset_path / "depictions.json"
                with open(path, 'w') as file:
                    json.dump(depictions, file)

    print(f"Successfully saved output to {path}")
