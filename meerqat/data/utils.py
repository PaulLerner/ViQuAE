# coding: utf-8

import hashlib


def md5(string: str) -> str:
    """Utility function. Uses hashlib to compute the md5 sum of a string.
    First encodes the string and utf-8.
    Lastly decodes the hash using hexdigest.
    """
    return hashlib.md5(string.encode("utf-8")).hexdigest()
