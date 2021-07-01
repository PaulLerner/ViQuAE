# coding: utf-8
from datasets import load_dataset
from pathlib import Path

from meerqat import __file__ as ROOT_PATH

DATA_ROOT_PATH = (Path(ROOT_PATH).parent.parent/"data").resolve()
KVQA_PATH = DATA_ROOT_PATH/"KVQA"
OKVQA_PATH = DATA_ROOT_PATH/"OK-VQA"
MSCOCO_PATH = DATA_ROOT_PATH/"MS-COCO"


def map_kilt_triviaqa():
    """As instructed by https://github.com/huggingface/datasets/blob/master/datasets/kilt_tasks/README.md"""

    kilt_tasks = load_dataset("kilt_tasks")
    # Most tasks in KILT already have all required data, but KILT-TriviaQA
    # only provides the question IDs, not the questions themselves.
    # Thankfully, we can get the original TriviaQA data with:
    trivia_qa = load_dataset('trivia_qa', 'unfiltered.nocontext')
    # The KILT IDs can then be mapped to the TriviaQA questions with:
    triviaqa_map = {}

    def add_missing_data(x, trivia_qa_subset, triviaqa_map):
        i = triviaqa_map[x['id']]
        x['input'] = trivia_qa_subset[i]['question']
        x['output']['original_answer'] = trivia_qa_subset[i]['answer']['value']
        return x

    for k in ['train', 'validation', 'test']:
        triviaqa_map = dict([(q_id, i) for i, q_id in enumerate(trivia_qa[k]['question_id'])])
        kilt_tasks[k + '_triviaqa'] = kilt_tasks[k + '_triviaqa'].filter(lambda x: x['id'] in triviaqa_map)
        kilt_tasks[k + '_triviaqa'] = kilt_tasks[k + '_triviaqa'].map(
            add_missing_data,
            fn_kwargs=dict(trivia_qa_subset=trivia_qa[k], triviaqa_map=triviaqa_map)
        )

    return kilt_tasks


def remove_special_fields(paragraphs):
    """N. B. this code puts a lot of trust into KILT pre-processing
    https://github.com/facebookresearch/KILT/blob/master/scripts/create_kilt_data_paragraphs.py
    and simply removes the title (1st paragraph), sections titles ("Section::::") and bullet-points ("BULLET::::")
    """
    preprocessed_paragraphs = []
    # drop title (first paragraph)
    for paragraph in paragraphs[1:]:
        # remove sections titles and bullet-points
        if paragraph.startswith("Section::::") or paragraph.startswith("BULLET::::"):
            continue
        # keep as is
        else:
            preprocessed_paragraphs.append(paragraph)
    return preprocessed_paragraphs


def paragraphs_preprocess(paragraphs, method=None, **kwargs):
    """
    Parameters
    ----------
    paragraphs: List[str]
        List of paragraphs to preprocess
    method: str, optional
        type of pre-processing, defaults to None (i.e. identity function)
    **kwargs: additional arguments are passed to the appropriate pre-processing function

    Returns
    -------
    paragraphs: List[str]

    """
    methods = {
        None: lambda paragraphs: paragraphs,
        "special_fields": remove_special_fields
    }
    return methods[method](paragraphs, **kwargs)


def uniform_passages(paragraphs, tokenizer, n=100):
    """
    Parameters
    ----------
    paragraphs: List[str]
        List of pre-processed paragraphs to split into passages

    Returns
    -------
    passages: List[str]
    """
    text = ''.join(paragraphs)
    raise NotImplementedError


def make_passages(paragraphs, method=None, preprocessing_method=None, preprocessing_kwargs={}, **kwargs):
    """
    Parameters
    ----------
    paragraphs: List[str]
        List of paragraphs to preprocess
    method: str, optional
        How to split the text in passages, defaults to keep the original paragraphs
    """
    paragraphs = paragraphs_preprocess(paragraphs, method=preprocessing_method, **preprocessing_kwargs)
    methods = {
        None: lambda paragraphs: paragraphs,
        "uniform": uniform_passages
    }
    return methods[method](paragraphs, **kwargs)