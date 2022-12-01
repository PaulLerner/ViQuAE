# coding: utf-8
"""
======
Usages
======
Gathers data about entities mentioned in questions via Wikidata, Wikimedia Commons SPARQL services and Wikimedia REST API.

You should run all of these in this order to get the whole cake:

-----------------
``data entities``
-----------------

**input/output**: ``entities.json`` (output of ``kilt2vqa.py count_entities``)  
queries many different attributes for all entities in the questions 

Also sets a 'reference image' to the entity using Wikidata properties in the following order of preference:
    - P18 ‘image’ (it is roughly equivalent to the infobox image in Wikipedia articles)
    - P154 ‘logo image’
    - P41 ‘flag image’
    - P94 ‘coat of arms image’
    - P2425 ‘service ribbon image’

-----------------
``data feminine``
-----------------
**input**: ``entities.json``  
**output**: ``feminine_labels.json``  
gets feminine labels for classes and occupations of these entities

---------------------
``data superclasses``
---------------------
**input**: ``entities.json``  
**output**: ``<n>_superclasses.json``  

gets the superclasses of the entities classes up ``n`` level (defaults to 'all', i.e. up to the root)

---------------------
Depictions (optional)
---------------------
we found that heuristics/images based on depictions were not that discriminative

^^^^^^^^^^^^^^^^^^^^^^^^^^
``commons sparql depicts``
^^^^^^^^^^^^^^^^^^^^^^^^^^
**input/output**: ``entities.json``  
Find all images in Commons that *depict* the entities

^^^^^^^^^^^^^^^^^^^^^^^^^^^
``commons sparql depicted``
^^^^^^^^^^^^^^^^^^^^^^^^^^^
**input**: ``entities.json``  
**output**: ``depictions.json``  
Find all entities depicted in the previously gathered step

^^^^^^^^^^^^^^^^^
``data depicted``
^^^^^^^^^^^^^^^^^
**input**: ``entities.json``, ``depictions.json``   
**output**: ``entities.json``  
Gathers the same data as in ``wiki.py data entities <subset>`` for *all* entities depicted in any of the depictions  
Then apply a heuristic to tell whether an image depicts the entity prominently or not:
*the depiction is prominent if the entity is the only one of its class*, e.g.:
    - *pic of Barack Obama and Joe Biden* -> not prominent  
    - *pic of Barack Obama and the Eiffel Tower* -> prominent  

Note this heuristic is not used in ``commons heuristics``


----------
``filter``
----------
**input/output**: ``entities.json``  
Filters entities w.r.t. to their class/nature/"instance of" and date of death, see ``wiki.py`` docstring for option usage (TODO share concrete_entities/abstract_entities)
Also entities with a ‘sex or gender’ (P21) or ‘occupation’ (P106) are kept by default.

Note this deletes data so maybe save it if you're unsure about the filter.

----------------
``commons rest``
----------------
**input/output**: ``entities.json``  

Gathers images and subcategories recursively from the entity root commons-category

Except if you have a very small dataset you should probably set ``--max_images=0`` to query only categories and use ``wikidump.py`` to gather images from those.  
``--max_categories`` defaults to 100.


----------------------
``commons heuristics``
----------------------
**input/output**: ``entities.json``  
Run ``wikidump.py`` first to gather images.  
Compute heuristics for the image (control with ``<heuristic>``, default to all):
    - ``categories``: the entity label should be included in *all* of the image category
    - ``description``: the entity label should be included in the image description
    - ``title``: the entity label should be included in the image title/file name
    - ``depictions``: the image should be tagged as *depicting* the entity (gathered in ``commons sparql depicts``)


==============
For ``docopt``
==============


Usage:
wiki.py data entities <subset> [--skip=<attribute>]
wiki.py data feminine <subset>
wiki.py data depicted <subset>
wiki.py data superclasses <subset> [--n=<n>]
wiki.py commons sparql depicts <subset>
wiki.py commons sparql depicted <subset>
wiki.py commons rest <subset> [--max_images=<max_images> --max_categories=<max_categories>]
wiki.py commons heuristics <subset> [<heuristic>...]
wiki.py filter <subset> [--superclass=<level> --positive --negative --deceased=<year> <classes_to_exclude>...]

Options:
--n=<n>                          Maximum level of superclasses. Defaults to all superclasses
--max_images=<n>                 Maximum number of images to query per entity/root category.
                                     Set to 0 if you only want to query categories [default: 1000].
--max_categories=<n>             Maximum number of categories to query per entity/root category [default: 100].
<heuristic>...                   Heuristic to compute for the image, one of {"categories", "description", "depictions", "title"}
                                    Defaults to all valid heuristics (listed above)
--superclass=<level>             Level of superclasses in the filter, int or "all" (defaults to None i.e. filter only classes)
--positive                       Keep only classes in "concrete_entities" + entities with gender (P21) or occupation (P106).
                                    Applied before negative_filter.
--negative                       Keep only classes that are not in "abstract_entities". Applied after positive_filter
--deceased=<year>                Remove humans (Q5) that are alive or deceased after <year> (might avoid trouble with GDPR)
<classes_to_exclude>...          Additional classes to exclude in the negative_filter (e.g. "Q5 Q82794")
                                    Note that you can use this option even without --negative i.e. specifying your own "abstract_entities"

=========
Functions
=========
"""
import re
import time
import json
import warnings
import datetime
from collections import Counter
from tabulate import tabulate
import pandas as pd

import requests
# TODO make it optional in requirements.txt
try:
    from SPARQLWrapper import SPARQLWrapper, JSON
except ModuleNotFoundError as e:
    JSON = "JSON"
    warnings.warn(f"ModuleNotFoundError: {e}")
from urllib.error import HTTPError
from urllib3.exceptions import MaxRetryError
from tqdm import tqdm
from docopt import docopt

from .loading import DATA_ROOT_PATH, COMMONS_PATH
from .utils import md5

# One client (user agent + IP) is allowed 60 seconds of processing time each 60 seconds
# https://www.mediawiki.org/wiki/Wikidata_Query_Service/User_Manual
WIKIDATA_COMPUTE_LIMIT = 60

QID_URI_PREFIX = "http://www.wikidata.org/entity/"
HUMAN = QID_URI_PREFIX + 'Q5'

SPECIAL_PATH_URI_PREFIX = "http://commons.wikimedia.org/wiki/Special:FilePath/"
SPECIAL_FILE_PATH_URI_PREFIX = SPECIAL_PATH_URI_PREFIX + "File:"
UPLOAD_URI_PREFIX = "http://upload.wikimedia.org/wikipedia/commons/"
VALID_DATE_TYPE = 'http://www.w3.org/2001/XMLSchema#dateTime'

# restrict media to be images handleable by PIL.Image (or convertible with Wikimedia thumbnails)
VALID_ENCODING = {"png", "jpg", "jpeg", "tiff", "gif", "svg", "tif", "pdf", "djvu"}

# used to make thumbnails in file_name_to_thumbnail
EXTENSIONS_PRE_AND_SUFFIXES = {
    "svg": ("", ".png"),
    "tif": ("lossy-page1-", ".jpg"),
    "tiff": ("lossy-page1-", ".jpg"),
    "pdf": ("page1-", ".jpg"),
    "djvu": ("page1-", ".jpg")
}

# rules of preferences over licenses, the higher the better (0 is reserved for missing values or other licenses)
LICENSES = {
    "CC0": 8,
    "PUBLIC DOMAIN MARK": 7,
    "PUBLIC DOMAIN": 7,
    "PDM": 7,
    "BY": 6,
    "BY-SA": 5,
    "BY-NC": 4,
    "BY-ND": 3,
    "BY-NC-SA": 2,
    "BY-NC-ND": 1
}

# Template for wikidata to query many different attributes of a list of entities
# should be used like
# >>> WIKIDATA_QUERY % "wd:Q76 wd:Q78579194 wd:Q42 wd:Q243"
# i.e. entity ids are space-separated and prefixed by 'wd:'
WIKIDATA_QUERY = """
SELECT ?entity ?entityLabel ?instanceof ?instanceofLabel ?commons ?image ?flag ?coat_of_arms ?logo ?service_ribbon ?occupation ?occupationLabel ?gender ?genderLabel ?freebase ?date_of_birth ?date_of_death ?taxon_rank ?taxon_rankLabel
{
  VALUES ?entity { %s }
  OPTIONAL{ ?entity wdt:P373 ?commons . }
  ?entity wdt:P31 ?instanceof .
  OPTIONAL { 
    ?entity wdt:P18 ?_image . 
    BIND(replace(wikibase:decodeUri(STR(?_image))," ","_") AS ?image)
  }
  OPTIONAL { 
    ?entity wdt:P41 ?_flag . 
    BIND(replace(wikibase:decodeUri(STR(?_flag))," ","_") AS ?flag)
  }
  OPTIONAL { 
    ?entity wdt:P94 ?_coat_of_arms . 
    BIND(replace(wikibase:decodeUri(STR(?_coat_of_arms))," ","_") AS ?coat_of_arms)
  }
  OPTIONAL { 
    ?entity wdt:P154 ?_logo . 
    BIND(replace(wikibase:decodeUri(STR(?_logo))," ","_") AS ?logo)
  }
  OPTIONAL { 
    ?entity wdt:P2425 ?_service_ribbon . 
    BIND(replace(wikibase:decodeUri(STR(?_service_ribbon))," ","_") AS ?service_ribbon)
  }
  OPTIONAL { ?entity wdt:P21 ?gender . }
  OPTIONAL { ?entity wdt:P106 ?occupation . }
  OPTIONAL { ?entity wdt:P646 ?freebase . }
  OPTIONAL { ?entity wdt:P569 ?date_of_birth . }
  OPTIONAL { ?entity wdt:P570 ?date_of_death . }
  OPTIONAL { ?entity wdt:P105 ?taxon_rank . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""
# get all feminine labels
WIKIDATA_FEMININE_QUERY = """
SELECT ?entity ?entity_female_label 
{
  VALUES ?entity { %s }
  ?entity wdt:P2521 ?entity_female_label .
  FILTER(LANG(?entity_female_label) = "en").
}
"""

# query super classes of a given class list
# use
# >>> WIKIDATA_SUPERCLASSES_QUERY % (qids, "wdt:P279+")
# to query all superclasses
WIKIDATA_SUPERCLASSES_QUERY = """
SELECT ?class ?classLabel ?subclassof ?subclassofLabel
WHERE 
{
  VALUES ?class { %s }.
  ?class %s ?subclassof.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

# see update_from_data
RESERVED_IMAGES = ['image', 'logo', 'flag', 'coat_of_arms', 'service_ribbon']
MULTIPLE_KEYS = {'instanceof', 'occupation'}.union(RESERVED_IMAGES)
UNIQUE_KEYS = {'entityLabel', 'gender', 'genderLabel', 'commons', 'freebase', 'date_of_birth', 'date_of_death', 'taxon_rank', 'taxon_rankLabel'}

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

# get all files or sub-categories in a Commons category
# use like
# >>> COMMONS_REST_LIST.format(cmtitle=<str including "Category:" prefix>, cmtype="subcat"|"file")
# e.g.
# >>> COMMONS_REST_LIST.format(cmtitle="Category:Barack Obama in 2004", cmtype="subcat")
COMMONS_REST_LIST = "https://commons.wikimedia.org/w/api.php?action=query&list=categorymembers&cmtitle={cmtitle}&cmprop=title|type&format=json&cmcontinue&cmlimit=max&cmtype={cmtype}"

# query images URL, categories and description
# use like
# >>> COMMONS_REST_TITLE.format(titles=<title1>|<title2>) including the "File:" prefix
# e.g.
# >>> COMMONS_REST_TITLE.format(titles="File:Barack Obama foreign trips.png|File:Women for Obama luncheon September 23, 2004.png")
COMMONS_REST_TITLE = "https://commons.wikimedia.org/w/api.php?action=query&titles={titles}&prop=categories|description|imageinfo&format=json&iiprop=url|extmetadata&clshow=!hidden"

VALID_IMAGE_HEURISTICS = {"categories", "description", "depictions", "title"}


def file_name_to_thumbnail(file_name, image_width=None):
    """
    get upload.wikimedia.org url from image file_name using the desired thumbnail width

    Parameters
    ----------
    file_name: str
        file name/title (without the "File:" prefix)
    image_width: int, optional
        desired thumbnail width in pixels for the image url
        Defaults to full-size
    """
    file_name = file_name.replace(" ", "_")
    file_hash = md5(file_name)
    if image_width is None:
        thumb, sized_name = "", ""
    else:
        thumb = "thumb/"
        extension = file_name.split('.')[-1].lower()
        prefix, suffix = EXTENSIONS_PRE_AND_SUFFIXES.get(extension, ("", ""))
        sized_name = f"/{prefix}{image_width:d}px-{file_name}{suffix}"
    url = f"{UPLOAD_URI_PREFIX}{thumb}{file_hash[0]}/{file_hash[:2]}/{file_name}{sized_name}"

    return url


def thumbnail_to_file_name(url, original=True):
    """
    Handles thumbnails and special file paths

    If original (default), Returns the original file-name, i.e. with the original extension and without any size specification introduced in file_name_to_thumbnail

    e.g. "https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/foo.tif/lossy-page1-469px-foo.tif.jpg" -> "foo.tif"

    else, the file name is returned as processed in file_name_to_thumbnail, i.e. with size specification etc.
    e.g. "https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/foo.tif/lossy-page1-469px-foo.tif.jpg" -> "lossy-page1-469px-foo.tif.jpg"

    This is irrelevant of course if the url is not a thumbnail
    """
    if url.startswith(SPECIAL_FILE_PATH_URI_PREFIX):
        return url[len(SPECIAL_FILE_PATH_URI_PREFIX):]
    elif url.startswith(SPECIAL_PATH_URI_PREFIX):
        return url[len(SPECIAL_PATH_URI_PREFIX):]
    elif url.startswith(UPLOAD_URI_PREFIX):
        file_name = url[len(UPLOAD_URI_PREFIX):]
        if file_name.startswith('thumb/'):
            file_name = file_name[len('thumb/a/a8/'):].split('/')
            if original:
                return file_name[0]
            else:
                return file_name[-1]
        else:
            return file_name[len('a/a8/'):]
    else:
        return url.split('/')[-1]


def bytes2dict(b):
    return json.loads(b.decode("utf-8"))


def get_license(image):
    """Get license short-name, upper-cased. Returns empty-string ('') if unavailable"""
    return image.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "").upper()


def license_score(image):
    """Gets license value, normalize it and return score (in LICENSES)"""
    license_ = get_license(image)
    # look for the kind of CC license (handles dashes)
    cc = re.match(r"CC[ -](BY\S*)[ -]\d\.\d", license_)
    if cc is not None:
        license_ = cc.group(1)
    return LICENSES.get(license_, 0)


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
    sparql.setUseKeepAlive()
    results, qids = [], []
    skip_total = 0
    # query only n qid at a time
    for i, qid in enumerate(tqdm(wikidata_ids, desc=description)):
        qids.append(prefix+qid)
        if (i + 1) % n == 0 or i == (len(wikidata_ids) - 1):
            sparql.setQuery(query % " ".join(qids))
            try:
                response = sparql.query()
            except HTTPError as e1:
                if str(e1.code).strip() == '429':
                    # HACK: sleep WIKIDATA_COMPUTE_LIMIT seconds to avoid 'HTTP Error 429: Too Many Requests'
                    time.sleep(WIKIDATA_COMPUTE_LIMIT)
                    # try one more time
                    try:
                        response = sparql.query()
                    except HTTPError as e2:
                        warnings.warn(f"HTTPError: {e2}\n"
                                      f"Query failed twice after waiting {WIKIDATA_COMPUTE_LIMIT}s in-between, "
                                      f"skipping the following QIDs:\n{qids}")
                        skip_total += len(qids)
                        qids = []
                        continue
                else:
                    warnings.warn(f"HTTPError: {e1}\nskipping the following QIDs:\n{qids}")
                    skip_total += len(qids)
                    qids = []
                    continue
            results += response.convert()['results']['bindings']
            qids = []

    print(f"Query succeeded! Got {len(results)} results, skipped {skip_total} QIDs")

    return results


def update_from_data(entities, skip=None):
    """Updates entities with info queried in from Wikidata"""

    # query Wikidata
    if skip is None:
        wikidata_ids = entities.keys()
    # skip all entities that already have the `skip` attribute
    else:
        wikidata_ids = [qid for qid in entities if skip not in entities[qid]]
    results = query_sparql_entities(WIKIDATA_QUERY, WIKIDATA_ENDPOINT, wikidata_ids,
                                    description="Querying Wikidata")
    # update entities with results
    for result in tqdm(results, desc="Updating entities"):
        qid = result['entity']['value'].split('/')[-1]
        # handle keys/attributes that are unique
        for unique_key in (UNIQUE_KEYS & result.keys()):
            # simply add or update the key/attribute
            entities[qid][unique_key] = result[unique_key]
        # handle keys/attributes that may be multiple
        for multiple_key in (MULTIPLE_KEYS & result.keys()):
            # create a new dict for this key/attribute so we don't duplicate data
            entities[qid].setdefault(multiple_key, {})
            # store corresponding label in the 'label' field
            result[multiple_key]['label'] = result.get(multiple_key + 'Label')
            # value (e.g. QID) of the attribute serves as key
            multiple_value = result[multiple_key]['value']
            entities[qid][multiple_key][multiple_value] = result[multiple_key]

    return entities


def set_reference_images(entities):
    """Set a reference image using RESERVED_IMAGES as order of preference if the entity has any available"""
    for entity in entities.values():
        # try to get illustrative image, fallback on other images if available
        # "image" is expected to be the first element of RESERVED_IMAGES
        for entity_image_key in RESERVED_IMAGES:
            entity_image = entity.get(entity_image_key)
            if entity_image is not None:
                encoding = None
                # HACK: pop 'type' and 'value' that might have been gathered
                # when we considered only a single illustrative image per entity
                entity_image.pop('type', None)
                entity_image.pop('value', None)

                # filter encodings
                for v in entity_image.values():
                    url = v.get('value')
                    if v is None:
                        continue
                    encoding = url.split('.')[-1].lower()
                    if encoding in VALID_ENCODING:
                        entity['reference_image'] = url
                        break
                if encoding in VALID_ENCODING:
                    break
            
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
    results = query_sparql_entities(COMMONS_DEPICTED_ENTITIES_QUERY,
                                    COMMONS_SPARQL_ENDPOINT,
                                    depictions.keys(), prefix="sdc:",
                                    description="Querying Wikimedia Commons")
    # update depictions with results
    for result in tqdm(results, desc="Updating depictions"):
        qid = result['commons_entity']['value'].split('/')[-1]
        depictions[qid].append(result["depicted_entity"]['value'])
    return depictions


def depiction_instanceof_heuristic(depictions, entities):
    for qid, entity in tqdm(entities.items(), desc="Applying 'instanceof' heuristic"):
        if 'instanceof' not in entity:
            continue
        instanceof = entity['instanceof'].keys()
        entity_depictions = entity.get("depictions", {})    
        for mid, depiction in entity_depictions.items():
            mid = mid.split('/')[-1]
            depiction["prominent_instanceof_heuristic"] = True
            # iterate over all other entities depicted in depiction
            for other_qid in depictions[mid]:
                other_qid = other_qid.split('/')[-1]
                # skip self
                if other_qid == qid:
                    continue
                other_entity = entities[other_qid]
                other_instanceof = other_entity.get('instanceof', {}).keys()
                # heuristic: the depiction is prominent if the entity is the only one of the same instance
                # e.g. pic of Barack Obama and Joe Biden -> not prominent
                #      pic of Barack Obama and the Eiffel Tower -> prominent
                if instanceof & other_instanceof:
                    depiction["prominent_instanceof_heuristic"] = False
                    break
    return entities


def keep_prominent_depictions(entities):
    for entity in entities.values():
        depictions = entity.get("depictions")
        if not depictions:
            continue
        # filter out non-prominent depictions
        entity["depictions"] = {mid: depiction for mid, depiction in depictions.items()
                                if depiction.get('prominent_instanceof_heuristic', False)}
    return entities


def request(query, session, tries=0, max_tries=2):
    """GET query via requests, handles exceptions and returns None if something went wrong"""
    response = None
    base_msg = f"Something went wrong when requesting for '{query}':\n"
    if tries >= max_tries:
        warnings.warn(f"{base_msg}Maximum number of tries ({max_tries}) exceeded: {tries}")
        return response
    try:
        response = session.get(query, headers={'User-Agent':'meerqat bot 0.1'})
    except requests.exceptions.ConnectionError as e:
        warnings.warn(f"{base_msg}requests.exceptions.ConnectionError: {e}")
    except MaxRetryError as e:
        warnings.warn(f"{base_msg}MaxRetryError: {e}")
    except OSError as e:
        warnings.warn(f"{base_msg}OSError: {e}")
    except Exception as e:
        warnings.warn(f"{base_msg}Exception: {e}")

    if response is not None and response.status_code != requests.codes.ok:
        if response.status_code == 429:
            time.sleep(int(response.headers.get("Retry-After", 1)))
            return request(query, session, tries+1, max_tries=max_tries)
        warnings.warn(f"{base_msg}status code: {response.status_code}")
        response = None

    return response


def query_commons_subcategories(category, categories, images,
                                max_images=1000, max_categories=100,
                                n_queried_categories=0):
    """Query all commons subcategories (and optionally images) from a root category recursively

    Parameters
    ----------
    category: str
        Root category
    categories: dict
        {str: bool}, True if the category has been processed
    images: dict
        {str: dict}, Key is the file title, gathers data about the image, see query_image
    max_images: int, optional
        Maximum number of images to query per entity/root category.
        Set to 0 if you only want to query categories (images dict will be left empty)
        Defaults to 1000
    max_categories: int, optional
        Maximum number of categories to query per entity/root category.
        Enforced via:
        - n_queried_categories if max_images > 0
        - len(categories) otherwise
        Defaults to 100
    n_queried_categories: int, optional
        Keeps track of the number of queried categories in order to enforce max_categories
        Should be equal to the number of True in categories
        Defaults to 0

    Returns
    -------
    categories, images: dict
        Same as input, hopefully enriched with new data
    """
    query = COMMONS_REST_LIST.format(cmtitle=category, cmtype="subcat|file")
    session = requests.Session()
    response = request(query, session)
    if not response:
        return categories, images
    results = bytes2dict(response.content).get('query', {}).get('categorymembers')
    if results is None:
        return categories, images

    # recursive call: query subcategories of the subcategories
    categories[category] = True
    n_queried_categories += 1
    todo = []
    for result in results:
        title = result['title']
        type_ = result["type"]
        # first query all files in the category before querying subcategories
        # except if max_images <= 0, then only query subcategories
        if type_ == "file" and max_images > 0:
            # avoid querying the same image again and again as the same image is often in multiple categories
            if title in images:
                continue
            encoding = title.split('.')[-1].lower()
            if encoding not in VALID_ENCODING:
                continue
            images[title] = query_image(title, session)
        elif type_ == "subcat":
            # avoid 1. to get stuck in a loop 2. extra processing:
            # skip already processed categories
            if title not in categories:
                todo.append(title)
                # and keep track of the processed categories
                categories[category] = False
    # return when we have enough images or categories
    if len(images) > max_images or \
        (max_images > 0 and n_queried_categories > max_categories) or \
        (max_images <= 0 and len(categories) > max_categories):
        return categories, images
    # else query all subcategories
    for title in todo:
        query_commons_subcategories(title, categories, images,
                                    max_images=max_images,
                                    max_categories=max_categories,
                                    n_queried_categories=n_queried_categories)
    return categories, images


def query_image(title, session):
    # query images URL, categories and description
    # note: it might be better to batch the query but when experimenting with
    # batch size as low as 25 I had to deal with 'continue' responses...
    query = COMMONS_REST_TITLE.format(titles=title)
    response = request(query, session)
    if not response:
        return None
    result = bytes2dict(response.content)['query']['pages']
    # get first (only) value
    result = next(iter(result.values()))
    imageinfo = result.get('imageinfo', [{}])[0]
    image_categories = [c.get('title') for c in result['categories']] if 'categories' in result else None
    # filter metadata
    extmetadata = imageinfo.get('extmetadata', {})
    extmetadata.pop('Categories', None)
    # TODO add some preference rules according to extmetadata["LicenseShortName"]
    # not sure how the description of an image is metadata but anyway, I fount it there...
    imageDescription = extmetadata.pop('ImageDescription', {})
    image = {
        "categories": image_categories,
        "url": imageinfo.get("url"),
        "description": imageDescription,
        "extmetadata": extmetadata
    }
    return image


def save_image(url, session):
    file_name = thumbnail_to_file_name(url, original=False)
    image_path = COMMONS_PATH / file_name
    if len(file_name) > len("Dr._Paul_R._Heyl,_Scientists_for_Uncle_Sam_awarded_Magellan_Gold_Medal._Dr._L.J._Briggs,_left,_Assistant_Director_and_Dr._Paul_R._Heyl,_Chief_of_the_Sound_Section_of_the_United_States_Bureau_of_Standards,_LCCN2016888398_(cropped).tiff"):
        return None

    if not image_path.exists():
        # request image
        response = request(requests.utils.quote(url, safe=':/'), session)
        if not response:
            return None
        # save if request went OK
        with open(image_path, 'wb') as file:
            file.write(response.content)

    return image_path


def update_from_commons_rest(entities, max_images=1000, max_categories=100):
    n_images, n_categories = [], []
    for entity in tqdm(entities.values(), desc="Updating entities from Commons"):
        # query only entities that appear in dataset (some may come from 'depictions')
        if entity['n_questions'] < 1 or "commons" not in entity:
            continue
        category = "Category:" + entity['commons']['value']
        # query all images in of entity Commons category and subcategories recursively
        categories, images = {}, {}
        query_commons_subcategories(category, categories, images, max_images, max_categories)
        entity['images'] = images
        entity['categories'] = categories
        n_images.append(len(images))
        n_categories.append(len(categories))
    print(f"{len(n_images)} entities out of {len(entities)} have a root Commons Category and questions in the dataset\n"
          f"Retrieved images:\n{pd.DataFrame(n_images).describe()}\nand categories:\n{pd.DataFrame(n_categories).describe()}")
    return entities


def special_path_to_file_name(special_path):
    """split url, add "File:" prefix and replace underscores with spaces"""
    return "File:"+special_path.split("/")[-1].replace('_', ' ')


def image_heuristic(entities, heuristics=VALID_IMAGE_HEURISTICS):
    invalid_heuristics = VALID_IMAGE_HEURISTICS - heuristics
    if invalid_heuristics:
        raise NotImplementedError(f"No heuristic was implemented for {invalid_heuristics}\n"
                                  f"Use one of {VALID_IMAGE_HEURISTICS}")

    # TODO named entity/link in description heuristic
    scores, best_scores = [], []
    for entity in tqdm(entities.values(), desc="Applying heuristics"):
        label = entity.get("entityLabel", {}).get("value")
        if not label or 'images' not in entity:
            continue
        label = label.lower()
        best_score = -1
        # get file names of the depictions (add "File:" prefix and replace underscores with spaces)
        if "depictions" in heuristics:
            depictions = {special_path_to_file_name(depiction["special_path"]["value"])
                          for depiction in entity.get("depictions", {}).values()}
        for title, image in entity['images'].items():
            image.setdefault("heuristics", {})

            # entity label should be included in all of the images categories
            if "categories" in heuristics and image.get("categories"):
                included = True
                for category in image['categories']:
                    if label not in category.lower():
                        included = False
                        break
                if included:
                    image["heuristics"]["categories"] = True

            # entity label should be included in the description
            if "description" in heuristics:
                description = image.get("description", "")
                if isinstance(description, dict):
                    description = description.get('value')
                if description and label in description.lower():
                    image["heuristics"]["description"] = True

            # image should be tagged as depicting (P180) the entity on Commons
            if "depictions" in heuristics and title in depictions:
                image["heuristics"]["depictions"] = True

            # entity label should be included in the image title
            if "title" in heuristics and label in title.lower():
                image["heuristics"]["title"] = True

            score = len(image["heuristics"])
            scores.append(score)
            if score > best_score:
                best_score = score
        best_scores.append(best_score)
    print(f"Applied heuristics {heuristics} on {len(entities)} entities/{len(scores)} images\n"
          f"\nEntity-level best scores stats:\n{pd.DataFrame(best_scores).describe()}"
          f"\nImage-level scores stats:\n{pd.DataFrame(scores).describe()}")
    return entities


def exclude_classes(entities, classes_to_exclude, superclasses={}):
    filtered_entities = {}
    for qid, entity in tqdm(entities.items(), desc="Filtering entities classes"):
        classes = entity.get('instanceof', {}).keys()
        # class should be excluded
        if classes & classes_to_exclude:
            continue
        exclude_super_class = False
        for class_ in classes:
            # superclass should be excluded
            if superclasses.get(class_, {}).keys() & classes_to_exclude:
                exclude_super_class = True
                break
        if exclude_super_class:
            continue
        # else keep entity/class
        filtered_entities[qid] = entity

    print(f"filtered {len(entities)} entities to {len(filtered_entities)}")
    return filtered_entities


def keep_classes(entities, classes_to_keep, superclasses={}, attributes_to_keep={"gender", "occupation"}):
    filtered_entities = {}
    for qid, entity in tqdm(entities.items(), desc="Filtering entities classes"):
        # keep all entities with an attribute in attributes_to_keep
        has_attribute = False
        for attribute in attributes_to_keep:
            if entity.get(attribute):
                has_attribute = True
                break
        if has_attribute:
            filtered_entities[qid] = entity
            continue

        # else keep entities with appropriate class or superclass
        classes = entity.get('instanceof', {}).keys()
        # class should be kept
        if classes & classes_to_keep:
            filtered_entities[qid] = entity
            continue
        for class_ in classes:
            # superclass should be kept
            if superclasses.get(class_, {}).keys() & classes_to_keep:
                filtered_entities[qid] = entity
                break

    print(f"filtered {len(entities)} entities to {len(filtered_entities)}")
    return filtered_entities


def iso2year(iso):
    """Handles negative dates"""
    if iso.startswith("-"):
        return - datetime.datetime.fromisoformat(iso[1:]).year
    return datetime.datetime.fromisoformat(iso).year


def remove_alive_humans(entities, year_threshold=float("inf")):
    filtered_entities = {}
    for qid, entity in tqdm(entities.items(), desc="Removing alive humans"):
        if HUMAN in entity.get('instanceof', {}).keys():
            date_of_death = entity.get("date_of_death", {})
            if date_of_death.get('datatype') != VALID_DATE_TYPE:
                continue
            year_of_death = iso2year(date_of_death['value'][:-1])
            if year_of_death > year_threshold:
                continue
        filtered_entities[qid] = entity

    print(f"filtered {len(entities)} entities to {len(filtered_entities)} using year_threshold={year_threshold}")
    return filtered_entities


def query_superclasses(entities, wikidata_superclasses_query, n_levels=None):
    if n_levels:
        level, levels = [], []
        for _ in range(n_levels):
            level.append("wdt:P279")
            levels.append("/".join(level))
        levels = "|".join(levels)
    else:
        levels = "wdt:P279+"

    wikidata_superclasses_query = wikidata_superclasses_query % ("%s", levels)
    # get all 'instanceof' i.e. all classes
    classes = {qid.split('/')[-1]: class_
               for entity in entities.values()
               for qid, class_ in entity.get('instanceof', {}).items()}
    # query all 'subclassof' i.e. all superclasses
    results = query_sparql_entities(wikidata_superclasses_query, WIKIDATA_ENDPOINT,
                                    classes.keys(), description="Querying superclasses")
    superclasses = {}
    for result in results:
        qid_uri = result["class"]["value"]
        superclasses.setdefault(qid_uri, {})
        subclassof = result["subclassof"]["value"]
        result["subclassof"]["label"] = result["subclassofLabel"]
        superclasses[qid_uri][subclassof] = result["subclassof"]
    return superclasses


def uri_to_qid(uri):
    return uri.split("/")[-1]


def uris_to_qids(uris):
    return {uri_to_qid(uri) for uri in uris}


def query_feminine_labels(entities):
    # 1. get all classes and occupations
    qids = set()
    for entity in entities.values():
        qids.update(uris_to_qids(entity.get("instanceof", {}).keys()))
        qids.update(uris_to_qids(entity.get("occupation", {}).keys()))

    # 2. query feminine labels of qids
    results = query_sparql_entities(WIKIDATA_FEMININE_QUERY, WIKIDATA_ENDPOINT,
                                    qids, description="Querying feminine labels")
    feminine_labels = {}
    for result in results:
        qid = result["entity"]["value"]
        feminine_label = result["entity_female_label"]["value"]
        feminine_labels.setdefault(qid, feminine_label)

    return feminine_labels


def stats(entities):
    """Simply count the # of field for every entity"""
    counter = Counter()
    for entity in entities.values():
        counter += Counter(entity.keys())
    return counter


def print_stats(entities):
    counter = stats(entities)
    print(tabulate([counter], headers='keys', tablefmt='latex'))


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    subset = args['<subset>']

    # load entities
    subset_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    path = subset_path / "entities.json"
    with open(path) as file:
        entities = json.load(file)
    depictions_path = subset_path / "depictions.json"

    # update from Wikidata or Wikimedia Commons
    if args['data']:
        if args['entities']:
            entities = update_from_data(entities, skip=args["--skip"])
            output = set_reference_images(entities)
            print_stats(output)
        elif args['feminine']:
            output = query_feminine_labels(entities)
            path = subset_path/"feminine_labels.json"
        elif args['depicted']:
            # load depictions
            with open(depictions_path) as file:
                depictions = json.load(file)
            depicted_entities = {qid.split('/')[-1]: {"n_questions": 0} 
                                 for depiction in depictions.values() 
                                 for qid in depiction}
            # query data about all depicted entities
            depicted_entities = update_from_data(depicted_entities)
            # update with the original entities data
            depicted_entities.update(entities)
            # apply "instance of" heuristic to tell if a depiction is prominent or not
            # note the result is saved in 'entities' as it is entity-dependent
            # (the same picture can be prominent for entity A but not for B and C)
            output = depiction_instanceof_heuristic(depictions, depicted_entities)
            print_stats(output)

        elif args['superclasses']:
            n_levels = int(args['--n']) if args['--n'] else None
            output = query_superclasses(entities, WIKIDATA_SUPERCLASSES_QUERY, n_levels=n_levels)
            path = subset_path / f"{n_levels if n_levels else 'all'}_superclasses.json"

    elif args['commons']:
        if args['sparql']:
            # find images that depict the entities
            if args['depicts']:
                output = update_from_commons_sparql(entities)
                print_stats(output)

            # find entities depicted in the images
            elif args['depicted']:
                # get depictions
                depictions = {depiction.split('/')[-1]: []
                              for entity in entities.values()
                              for depiction in entity.get("depictions", {})}
                output = query_depicted_entities(depictions)
                path = depictions_path
                print_stats(output)
        elif args['rest']:
            max_images = int(args['--max_images'])
            max_categories = int(args['--max_categories'])
            output = update_from_commons_rest(entities, max_images, max_categories)
            print_stats(output)
        elif args['heuristics']:
            heuristics = set(args['<heuristic>']) if args['<heuristic>'] else VALID_IMAGE_HEURISTICS
            output = image_heuristic(entities, heuristics)
            print_stats(output)
    elif args['filter']:
        positive_filter = args['--positive']
        negative_filter = args['--negative']
        deceased = int(args['--deceased']) if args['--deceased'] else None
        superclass_level = args['--superclass']
        if superclass_level and superclass_level != "all":
            superclass_level = int(superclass_level)
        classes_to_exclude = set(QID_URI_PREFIX + qid for qid in args['<classes_to_exclude>'])
        if superclass_level:
            with open(subset_path / f"{superclass_level}_superclasses.json") as file:
                superclasses = json.load(file)
        else:
            superclasses = {}
        if positive_filter:
            with open(DATA_ROOT_PATH / "concrete_entities.csv") as file:
                classes_to_keep = set(line.split(",")[0] for line in file.read().split("\n")[1:] if line != '')
            entities = keep_classes(entities, classes_to_keep, superclasses)
        if negative_filter:
            with open(DATA_ROOT_PATH / "abstract_entities.csv") as file:
                abstract_entities = set(line.split(",")[0] for line in file.read().split("\n")[1:] if line != '')
            classes_to_exclude.update(abstract_entities)

        if classes_to_exclude:
            entities = exclude_classes(entities, classes_to_exclude, superclasses)
        if deceased is not None:
            entities = remove_alive_humans(entities, year_threshold=deceased)
        output = entities
        print_stats(output)

    # save output
    with open(path, 'w') as file:
        json.dump(output, file)

    print(f"Successfully saved output to {path}")
