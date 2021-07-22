import unittest

from meerqat.ir.search import map_indices


class MappingTester(unittest.TestCase):
    def test_basic(self):
        mapping = {
            10: [0, 1],
            1000: [2, 3, 4]
        }
        scores_batch = [[100, 10000]]
        indices_batch = [[10, 1000]]
        out_scores, out_indices = map_indices(scores_batch, indices_batch, mapping)
        self.assertEqual(out_scores, [[100, 100, 10000, 10000, 10000]], 'wrong scores')
        self.assertEqual(out_indices, [[0, 1, 2, 3, 4]])

    def test_k(self):
        mapping = {
            10: [0, 1],
            1000: [2, 3, 4]
        }
        scores_batch = [[100, 10000]]
        indices_batch = [[10, 1000]]
        out_scores, out_indices = map_indices(scores_batch, indices_batch, mapping, k=2)
        self.assertEqual(out_scores, [[100, 100]], 'wrong scores')
        self.assertEqual(out_indices, [[0, 1]])


if __name__ == '__main__':
    unittest.main()
