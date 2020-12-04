# coding: utf-8
from collections import Counter
import matplotlib.pyplot as plt
import spacy
from pathlib import Path

import meerqat.__file__ as ROOT_PATH


def count_categories(questions):
    """
    Parameters
    ----------
    questions: List[Doc]
        List of spacy Doc (POS-tagged)

    Returns
    -------
    categories: Counter
    """
    categories = Counter()
    for question in questions:
        first = question[0].lower_
        if first.startswith('wh'):
            category = first
        elif first.startswith('how'):
            category = f"{question[0]} [{question[1].pos_}]"
        elif first.startswith('in'):
            category = str(question[:2])
        else:
            category = f"[{question[0].pos_}] [{question[1].pos_}]"
        category = category.lower()
        categories[category] += 1
    return categories


def plot_categories(categories):
    """Sort then make a pie chart of categories"""
    vals, labels = [], []
    for k, v in sorted(categories.items()):
        labels.append(k)
        vals.append(v)

    plt.pie(vals, labels=labels, autopct="%.0f")


def load_data(data_path):
    """TODO: load the proper dataset format"""
    raise NotImplementedError("All you have to do is load the data ;)")
    return dataset


if __name__ == '__main__':
    dataset = load_data(Path(ROOT_PATH).parent/"data"/"dataset.XXX")
    # load spacy model (for POS tagging)
    model = spacy.load("en_core_web_sm")
    questions = [model(question.strip()) for question in dataset]
    categories = count_categories(questions)
    plot_categories(categories)
