# coding: utf-8

import hashlib

import pandas as pd


def md5(string: str) -> str:
    """Utility function. Uses hashlib to compute the md5 sum of a string.
    First encodes the string and utf-8.
    Lastly decodes the hash using hexdigest.
    """
    return hashlib.md5(string.encode("utf-8")).hexdigest()


def json_integer_keys(dictionary):
    """
    Convert all keys of the dictionay to an integer
    (so make sure all of the keys can be casted as integers and remain unique before using this)
    """
    return {int(k): v for k, v in dictionary.items()}


def to_latex(metrics):
    table = pd.DataFrame([metrics])*100
    return table.to_latex(float_format='%.1f')