import unittest

import numpy as np

from meerqat.ir.search import scores2dict, dict_batch2scores


class Dict2ScoresTester(unittest.TestCase):
    """Applies scores2dict and dict_batch2scores reciprocally to make sure we fall back on identity"""
    def test_batch(self):
        k = 10
        # this is sorted in desc order
        scores_input = np.arange(k*3, 0, -1).reshape(3, k)
        dict_batch = scores2dict(scores_input, scores_input)
        scores_output = np.array(dict_batch2scores(dict_batch, k=k))
        self.assertTrue((scores_input==scores_output).all())


if __name__ == '__main__':
    unittest.main()
