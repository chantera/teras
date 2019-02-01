import pathlib
import re
import sys
import unittest

import numpy as np
from teras.dataset.loader import CorpusLoader
from teras.io.reader import ConllReader
from teras.preprocessing import text


class DataLoader(CorpusLoader):

    def __init__(self,
                 word_embed_size=100,
                 pos_embed_size=100,
                 word_embed_file=None,
                 word_preprocess=lambda x: re.sub(r'[0-9]', '0', x.lower()),
                 word_unknown="UNKNOWN",
                 embed_dtype='float32'):
        super().__init__(reader=ConllReader())
        self.use_pretrained = word_embed_file is not None
        self.add_processor(
            'word', embed_file=word_embed_file, embed_size=word_embed_size,
            embed_dtype=embed_dtype, initializer=text.random_normal,
            preprocess=word_preprocess, unknown=word_unknown)
        self.add_processor(
            'pos', embed_file=None, embed_size=pos_embed_size,
            embed_dtype=embed_dtype, initializer=text.random_normal,
            preprocess=lambda x: x.lower())
        self.label_map = text.Vocab()
        self._sentences = {}

    def map(self, item):
        # item -> (words, postags, (heads, labels))
        words, postags, heads, labels = \
            zip(*[(token['form'],
                   token['postag'],
                   token['head'],
                   self.label_map.add(token['deprel']))
                  for token in item])
        sample = (self._word_transform(words),
                  self.get_processor('pos').fit_transform(postags),
                  (np.array(heads, dtype=np.int32),
                   np.array(labels, dtype=np.int32)))
        self._sentences[hash(tuple(sample[0]))] = words
        self._count += 1
        sys.stderr.write("\rload samples: {}".format(self._count))
        sys.stderr.flush()
        return sample

    def load(self, file, train=False, size=None, bucketing=False):
        self._count = 0
        if train and not self.use_pretrained:
            # assign an index if the given word is not in vocabulary
            self._word_transform = self.get_processor('word').fit_transform
        else:
            # return the unknown word index if the word is not in vocabulary
            self._word_transform = self.get_processor('word').transform
        return super().load(file, train, size, bucketing)

    def get_sentence(self, word_ids, default=None):
        return self._sentences.get(hash(tuple(word_ids)), default)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_sentences'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


SAMPLE_DIR = (pathlib.Path(__file__).parent / '../samples').resolve()
TRAIN_FILE = SAMPLE_DIR / 'wsj-train.conll'
TEST_FILE = SAMPLE_DIR / 'wsj-test.conll'


class TestLoader(unittest.TestCase):

    def setUp(self):
        self.loader = DataLoader(word_embed_file=None,
                                 word_embed_size=100,
                                 pos_embed_size=50)

    def test_load(self):
        train_dataset = self.loader.load(TRAIN_FILE, train=True, size=1000)
        sys.stderr.write("\n")
        sys.stderr.flush()
        test_dataset = self.loader.load(TEST_FILE, train=False, size=100)
        sys.stderr.write("\n")
        sys.stderr.flush()
        self.assertTrue(len(train_dataset) > 0)
        self.assertTrue(len(test_dataset) > 0)


if __name__ == "__main__":
    unittest.main()
