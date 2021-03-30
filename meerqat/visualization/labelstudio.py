# coding: utf-8
"""Usage: labelstudio.py [<path>...]"""

import json
from docopt import docopt
from tqdm import tqdm
from pathlib import Path

from meerqat.data.labelstudio import retrieve_vqa
from meerqat.visualization.kilt2vqa import HTML_FORMAT, TD_FORMAT

DISCARD_FORMAT = "&#9888; DISCARD: {reason}"


def write_html(completions):
    tds = []
    for completion_path in tqdm(completions):
        completion_path = Path(completion_path)
        with open(completion_path, 'r') as file:
            completion = json.load(file)
        vqa = retrieve_vqa(completion)
        discard = vqa.pop("discard", None)
        if discard is None:
            vq = vqa["vq"]
        else:
            vq = DISCARD_FORMAT.format(reason=discard)
        td = TD_FORMAT.format(
            original_question=vqa['question'],
            url=vqa['image'],
            generated_question=vq,
            answer=vqa['answer'],
            qid=vqa['wikidata_id']
        )
        tds.append(td)
    html = HTML_FORMAT.format(tds="\n".join(tds))
    with open(completion_path.parent.parent / 'vqa.html', 'w') as file:
        file.write(html)


def main():
    args = docopt(__doc__)
    completions = args['<path>']
    write_html(completions)


if __name__ == '__main__':
    main()
