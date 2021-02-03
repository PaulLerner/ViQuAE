# coding: utf-8
"""Usage:
okvqa.py <subset> [--n=<n>]

Options:
    --n=<n>         Number of examples to visualize [default: 50].
"""

import numpy as np
import json
from docopt import docopt
from tqdm import tqdm

from meerqat.data.loading import DATA_ROOT_PATH, OKVQA_PATH, MSCOCO_PATH
from meerqat.data.okvqa import JPG_FORMAT


# HTML document format
HTML_FORMAT = """
<html>
<head>
    <link rel="stylesheet" href="styles.css">
</head>
<table>
    <tr>
        <th>Image</th>
        <th>Question</th>
        <th>Answers</th>
    </tr>
    {tds}
</table>
</html>
"""
TD_FORMAT = """
<tr>
    <td><img src="{path}" width="400"></td>
    <td>{question}</td>
    <td>{answers}</td>
</tr>
"""


def write_html(questions, annotations, visualization_path, subset, n=50):
    tds = []
    indices = np.arange(n)
    np.random.shuffle(indices)
    for i in tqdm(indices):
        question = questions[i]
        annotation = annotations[i]
        assert question['question_id'] == annotation['question_id']
        assert question['image_id'] == annotation['image_id']
        answers = []
        for answer, count in annotation['answer_counter'].items():
            answers.append(f"{answer} ({count['count']})")
        image_path = MSCOCO_PATH / JPG_FORMAT.format(subset=subset, image_id=question['image_id'])
        td = TD_FORMAT.format(
                path=image_path,
                question=question['question'],
                answers=",".join(answers)
            )
        tds.append(td)
    html = HTML_FORMAT.format(tds="\n".join(tds))
    with open(visualization_path/'okvqa.html', 'w') as file:
        file.write(html)


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    subset = args['<subset>']
    n = int(args['--n'])
    visualization_path = DATA_ROOT_PATH / "visualization" / "OK-VQA" / subset
    visualization_path.mkdir(exist_ok=True, parents=True)

    # load data
    with open(OKVQA_PATH / f"meerqat_OpenEnded_mscoco_{subset}2014_questions.json") as file:
        questions = json.load(file)
    annotations_file_name = f"meerqat_mscoco_{subset}2014_annotations.json"
    with open(OKVQA_PATH / annotations_file_name) as file:
        annotations = json.load(file)
    assert len(questions['questions']) == len(annotations['annotations'])

    # write result to HTML for visualization
    write_html(questions['questions'], annotations['annotations'], visualization_path, subset, n=n)

