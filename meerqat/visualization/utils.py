# coding: utf-8
from tabulate import tabulate
import numpy as np


def simple_stats(values, tablefmt="simple"):
    """Computes

    Parameters
    ----------
    values: 1D array-like
    tablefmt: table format used in tabulate

    Returns
    -------
    stats: str
        tabulated stats using tabulate
    """
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    value_stats = [values.sum(), np.mean(values)]
    value_stats += np.quantile(values, [0., 0.25, 0.5, 0.75, 1.0]).tolist()
    headers = ["sum", "mean", "min", "1st quartile", "median", "3rd quartile", "max"]
    return tabulate([value_stats], headers=headers, tablefmt=tablefmt)


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
