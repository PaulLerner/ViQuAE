"""Usage: stats.py <dataset> <key>... [--len --output=<path> --kwargs=<path> --pgf]

Options:
--len           Compute stats about the lengths of the lists instead of the values themselves
"""
import json
from docopt import docopt
from pathlib import Path

import pandas as pd
import seaborn as sns

from datasets import load_from_disk, DatasetDict


def dataset_stats(dataset, keys, output_path=None, lengths=False, pgf=False, discrete=None, **kwargs):
    discrete = lengths if discrete is None else discrete
    for key in keys:
        print(key)
        values = dataset[key]
        if isinstance(values[0], list):
            # compute stats about the lengths of the lists instead of the values themselves
            if lengths:
                # None counts as 0-length
                values = [len(x) if x is not None else 0 for x in values]
            # simply flatten
            else:
                values = [x_i for x in values for x_i in x]
        df = pd.DataFrame(values, columns=[key])
        print(df.describe().to_latex())
        print("equal zero:", (df == 0).sum())
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)
            fig = sns.displot(data=df, x=key, discrete=discrete, **kwargs)
            fig.savefig(output_path/f"{key}_distribution.png")
            if pgf:
                fig.savefig(output_path/f"{key}_distribution.pgf")
        print('\n*******************\n')


def main(dataset, keys, output_path=None, **kwargs):
    if isinstance(dataset, DatasetDict):
        for name, subset in dataset.items():
            subset_output_path = output_path/name if output_path is not None else None
            print(f"Statistics of the {name} set:")
            dataset_stats(subset, keys, output_path=subset_output_path, **kwargs)
            print('\n===================\n===================\n')
    else:
        dataset_stats(dataset, keys, output_path=output_path, **kwargs)


if __name__ == '__main__':
    args = docopt(__doc__)
    dataset = load_from_disk(args['<dataset>'])
    if args['--kwargs'] is not None:
        with open(args['--kwargs'], 'rt') as file:
            kwargs = json.load(file)
    else:
        kwargs = {}
    output_path = Path(args['--output']) if args['--output'] is not None else None
    main(dataset, args['<key>'], output_path=output_path, lengths=args['--len'], pgf=args['--pgf'], **kwargs)