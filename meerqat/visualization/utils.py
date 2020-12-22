# coding: utf-8
from tabulate import tabulate


def viz_spacy(doc, attrs=["pos_", "dep_"]):
    """
    Parameters
    ----------
    doc: spacy Doc
    attrs: List
        Attributes needed for visualization,
        defaults to POS and dependency tag.

    Returns
    -------
    string: str
        Input document tabulated with a line for each doc attributes (tokens, POS, ...)
    """
    if isinstance(doc, dict):
        viz = []
        text = doc["text"]
        for token in doc['tokens']:
            start, end = token['start'], token['end']
            v_token = [text[start: end]]
            for attr in attrs:
                v_token.append(token[attr[:-1]])
            ent_type = ""
            for ent in doc["ents"]:
                if start >= ent["start"] and end <= ent["end"]:
                    ent_type = ent['label']
            v_token.append(ent_type)
            viz.append(v_token)
    else:
        attrs = ["text", "ent_type_"] + attrs
        viz = [[getattr(token, attr) for attr in attrs] for token in doc]
    return tabulate(zip(*viz))
