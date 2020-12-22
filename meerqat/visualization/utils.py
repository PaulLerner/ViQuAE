# coding: utf-8
from tabulate import tabulate


def viz_spacy(doc, attrs=["text", "pos_", "dep_", "ent_type_"]):
    """
    Parameters
    ----------
    doc: spacy Doc
    attrs: List
        Attributes needed for visualization,
        defaults to tokens, POS, dependency tag and entity type.

    Returns
    -------
    string: str
        Input document tabulated with a line for each doc attributes (tokens, POS, ...)
    """
    viz = [[getattr(token, attr) for attr in attrs] for token in doc]
    return tabulate(zip(*viz))
