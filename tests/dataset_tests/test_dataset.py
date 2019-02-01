import sys
import unittest

import numpy as np
from teras.dataset import Dataset, BucketDataset


class TestDataset(unittest.TestCase):

    def setUp(self):
        N_SAMPLES = 1500
        MAX_LENGTH = 128
        N_VOCAB = 5000
        N_TAGS = 64

        samples = []
        for index in range(N_SAMPLES):
            length = np.random.choice(MAX_LENGTH)
            words = np.random.choice(N_VOCAB, length)
            tags = np.random.choice(N_TAGS, length)
            samples.append((words, tags, index))
        self.samples = samples

    def test_batch(self):
        dataset = Dataset(self.samples)
        for i, batch in enumerate(
                dataset.batch(size=32, shuffle=False, colwise=False)):
            self.assertTrue((i < 46 and len(batch) == 32)
                            or (i == 46 and len(batch) == 28))
            self.assertTrue(all(sample[2] == i * 32 + j
                                for j, sample in enumerate(batch)))
        for i, batch in enumerate(
                dataset.batch(size=32, shuffle=True, colwise=False)):
            self.assertTrue(any(sample[2] != i * 32 + j
                                for j, sample in enumerate(batch)))
        for i, batch in enumerate(
                dataset.batch(size=32, shuffle=False, colwise=True)):
            self.assertTrue(len(batch) == 3)
            self.assertTrue((i < 46 and len(batch[0]) == 32)
                            or (i == 46 and len(batch[0]) == 28))
            self.assertTrue(all(index == i * 32 + j
                                for j, index in enumerate(batch[2])))
        for i, batch in enumerate(
                dataset.batch(size=32, shuffle=True, colwise=True)):
            self.assertTrue(any(index != i * 32 + j
                                for j, index in enumerate(batch[2])))

    def test_bucketing(self):
        dataset = BucketDataset(self.samples, key=0, equalize_by_key=False)
        batch_min_max_list = []
        tail = None
        for i, batch in enumerate(
                dataset.batch(size=32, shuffle=True, colwise=False)):
            lengths = [len(sample[0]) for sample in batch]
            batch_size = len(lengths)
            if batch_size == 28:
                self.assertTrue(tail is None)
                tail = i
            else:
                self.assertTrue(batch_size == 32)
            batch_min_max = min(lengths), max(lengths)
            batch_min_max_list.append(batch_min_max)
        self.assertTrue(len(batch_min_max_list) == 47)
        batch_min_max_list.sort(key=lambda x: x[0] * 10 + x[1])
        print("min_max: {}".format(batch_min_max_list), file=sys.stderr)
        for i in range(1, len(batch_min_max_list)):
            self.assertTrue(batch_min_max_list[i][0]
                            >= batch_min_max_list[i - 1][1])

    def test_averaged_bucketing(self):
        dataset = BucketDataset(self.samples, key=0, equalize_by_key=True)
        n_words_sum = 0
        batch_min_max_list = []
        tail = None
        for i, batch in enumerate(
                dataset.batch(size=1000, shuffle=True, colwise=False)):
            lengths = [len(sample[0]) for sample in batch]
            n_words = sum(lengths)
            if n_words < 1000:
                self.assertTrue(tail is None)
                tail = i
            else:
                self.assertTrue(n_words < 1000 + lengths[-1])
            batch_min_max = min(lengths), max(lengths)
            batch_min_max_list.append(batch_min_max)
            n_words_sum += n_words
        self.assertTrue(len(batch_min_max_list) <= np.ceil(n_words_sum / 1000))
        batch_min_max_list.sort(key=lambda x: x[0] * 10 + x[1])
        for i in range(1, len(batch_min_max_list)):
            self.assertTrue(batch_min_max_list[i][0]
                            >= batch_min_max_list[i - 1][1])


if __name__ == "__main__":
    unittest.main()
