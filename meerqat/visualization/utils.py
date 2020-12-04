# coding: utf-8
from tabulate import tabulate


def viz_pos(doc):
    """
    Parameters
    ----------
    doc: spacy Doc

    Returns
    -------
    string: str
        Input document tabulated with a line for the tokens and another for POS
    """
    pos = [(token.text, token.pos_) for token in doc]
    return tabulate(zip(*pos))
