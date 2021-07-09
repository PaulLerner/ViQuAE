import unittest

from collections import Counter

from meerqat.ir.metrics import compute_metrics


class HitsAtKTester(unittest.TestCase):
    def test_no_match(self, metrics=Counter(), K=100, ks=[1, 5, 10, 20, 100]):
        retrieved_batch = [[1]]
        relevant_batch = [[0]]
        compute_metrics(metrics, retrieved_batch, relevant_batch, K=K, ks=ks)
        for k in ks:
            self.assertEqual(metrics[f'hits@{k}'], 0)

    def test_one_match(self, metrics=Counter(), K=100, ks=[1, 5, 10, 20, 100]):
        retrieved_batch = [[1]]
        relevant_batch = [[1]]
        compute_metrics(metrics, retrieved_batch, relevant_batch, K=K, ks=ks)
        for k in ks:
            self.assertEqual(metrics[f'hits@{k}'], 1)

    def test_ks(self, metrics=Counter(), K=50, ks=[1, 5, 10, 20, 100]):
        retrieved_batch = [[0, 2]]
        relevant_batch = [[1, 2]]
        compute_metrics(metrics, retrieved_batch, relevant_batch, K=K, ks=ks)
        self.assertEqual(metrics['hits@1'], 0)
        self.assertEqual(metrics['hits@5'], 1)
        self.assertNotIn('hits@100', metrics)
        self.assertIn('hits@20', metrics)

    def test_batch(self, metrics=Counter(), K=100, ks=[1, 5, 10, 20, 100]):
        retrieved_batch = [[1], [1]]
        relevant_batch = [[1], [1]]
        compute_metrics(metrics, retrieved_batch, relevant_batch, K=K, ks=ks)
        for k in ks:
            self.assertEqual(metrics[f'hits@{k}'], 2)


if __name__ == '__main__':
    unittest.main()