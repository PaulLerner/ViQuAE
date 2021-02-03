# coding: utf-8
"""Usage:
okvqa.py <subset> [--threshold=<threshold> <invalid_ent_type>...]

Options:
--threshold=<threshold>     Minimum number of people who answered that [default: 1].
"""
import spacy
import json
from collections import Counter

from tqdm import tqdm
from docopt import docopt

from meerqat.data.loading import OKVQA_PATH


JPG_FORMAT = "COCO_{subset}2014_{image_id:012d}.jpg"


def keep_entity_questions(questions, annotations, counter_threshold=1, invalid_ent_types=set()):
    entity_questions, entity_annotations = [], []
    for question, annotation in tqdm(zip(questions, annotations)):
        assert question['question_id'] == annotation['question_id']
        assert question['image_id'] == annotation['image_id']
        for answer in annotation['answer_counter'].values():
            ents = [ent for ent in answer['spacy']['ents'] if ent['label'] not in invalid_ent_types]
            if ents and answer['count'] >= counter_threshold:
                entity_questions.append(question)
                entity_annotations.append(annotation)
                break

    return entity_questions, entity_annotations


def run_spacy(annotations):
    model = spacy.load("en_core_web_lg")
    for annotation in tqdm(annotations):
        answer_counter = Counter(answer['answer'] for answer in annotation['answers'])
        for answer, count in answer_counter.items():
            answer_counter[answer] = {'count': count, 'spacy': model(answer).to_json()}
        annotation['answer_counter'] = answer_counter
    return annotations


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    subset = args['<subset>']
    counter_threshold = int(args['--threshold'])
    invalid_ent_types = set(args['<invalid_ent_type>'])
    questions_file_name = f"OpenEnded_mscoco_{subset}2014_questions.json"
    with open(OKVQA_PATH/questions_file_name) as file:
        questions = json.load(file)
    annotations_file_name = f"mscoco_{subset}2014_annotations.json"
    with open(OKVQA_PATH/annotations_file_name) as file:
        annotations = json.load(file)
    assert len(questions['questions']) == len(annotations['annotations'])
    annotations['annotations'] = run_spacy(annotations['annotations'])
    questions['questions'], annotations['annotations'] = keep_entity_questions(questions['questions'],
                                                                               annotations['annotations'],
                                                                               counter_threshold,
                                                                               invalid_ent_types)
    with open(OKVQA_PATH/f"meerqat_{questions_file_name}", "w") as file:
        json.dump(questions, file)
    with open(OKVQA_PATH/f"meerqat_{annotations_file_name}", "w") as file:
        json.dump(annotations, file)

