# coding: utf-8
"""
**input/output**: ``entities.json``  
Parses the dump (should be downloaded first, TODO add instructions), gathers images and assign them to the relevant entity given its common categories (retrieved in ``wiki.py commons rest``)  
Note that the wikicode is parsed very lazily and might need a second run depending on your application, e.g. templates are not expanded...

Usage: wikidump.py <subset>
"""
import bz2
import xml.etree.ElementTree as ET
from tqdm import tqdm
from docopt import docopt
import json
import re
import pandas as pd

from .loading import DATA_ROOT_PATH
from .wiki import VALID_ENCODING


NAMESPACE = {"mw": "http://www.mediawiki.org/xml/export-0.10/"}


def parse_file(path):
    if path.suffix == ".bz2":
        with bz2.open(path, "rb") as file:
            tree = ET.parse(file)
    else:
        tree = ET.parse(path)
    return tree


def find(element, tag, namespace=NAMESPACE):
    """test if element is None before returning ET.Element.find"""
    if element is None:
        return None
    return element.find(tag, namespace)


def find_text(element, tag, namespace=NAMESPACE):
    """returns result.text if result is not None"""
    result = find(element, tag, namespace)
    if result is None:
        return None
    return result.text


def get_field(wikitext, image, field):
    result = re.findall(rf"{field}=\s*(.+)\n", wikitext)
    if result:
        image[field.lower()] = result[0]
    return result


def process_article(article, entities, entity_categories):
    for page in article:
        title = find_text(page, "mw:title")
        # keep only files with valid encoding
        if title is None or not title.startswith("File:") or title.split('.')[-1].lower() not in VALID_ENCODING:
            continue

        revision = find(page, "mw:revision")
        if revision is None:
            continue
        wikitext = find_text(revision, "mw:text")
        if wikitext is None:
            continue

        # find categories
        categories = set()
        for internal_link in re.findall("\[\[(.+)\]\]", wikitext):
            if internal_link.lower().startswith("category:"):
                # remove name from link
                name = internal_link.find("|")
                if name >= 0:
                    internal_link = internal_link[: name]
                # make "Category" sentence-cased
                categories.add("C"+internal_link[1: ])
        # is there any entity with these categories?
        # note this also filters in case we did not find any category in wikitext
        if not (categories & entity_categories):
            continue

        image = {"categories": list(categories),
                 "timestamp": find_text(revision, "mw:timestamp")}
        contributor = find(revision, "mw:contributor")
        image["username"] = find_text(contributor, "mw:username")
        for field in ["Date", "Author"]:
            get_field(wikitext, image, field)

        description = re.search(r"description\s*=\s*(.+)", wikitext, flags=re.IGNORECASE|re.DOTALL|re.MULTILINE)
        if description is not None:
            description = description.group(1)
            i_new_field = description.find("\n|")
            if i_new_field >= 0:
                description = description[:i_new_field]
        image["description"] = description

        for license_match in re.finditer(r"{{int:license-header}}\s*=+", wikitext):
            license_ = re.findall("{{.+}}", wikitext[license_match.end():])
            if license_:
                image["license"] = license_[0]
            break

        # find entities with appropriate categories and save the image
        for entity in entities.values():
            if entity["n_questions"] < 1:
                continue
            if entity.get("categories", {}).keys() & categories:
                entity.setdefault("images", {})
                entity["images"][title] = image

    return entities


def process_articles(dump_path, entities):
    # set of all categories to enable faster search
    categories = {category for entity in entities.values() if entity["n_questions"] > 0
                           for category in entity.get("categories", {})}
    articles_path = list(dump_path.glob(r"commonswiki-latest-pages-articles[0-9]*"))
    for article_path in tqdm(articles_path, desc="Processing articles"):
        article = parse_file(article_path).getroot()
        process_article(article, entities, categories)
    return entities


if __name__ == "__main__":
    # parse arguments
    args = docopt(__doc__)
    subset = args['<subset>']

    # load entities
    subset_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    path = subset_path / "entities.json"
    with open(path, 'r') as file:
        entities = json.load(file)

    dump_path = DATA_ROOT_PATH / "commonswiki"
    process_articles(dump_path, entities)

    # save output
    with open(path, 'w') as file:
        json.dump(entities, file)

    print(f"Successfully saved output to {path}")

    n_images = [len(entity.get('images', [])) for entity in entities.values()]
    print(f"Gathered images from {len(entities)} entities:\n{pd.DataFrame(n_images).describe()}")
