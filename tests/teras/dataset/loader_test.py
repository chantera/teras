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
        super(DataLoader, self).__init__(reader=ConllReader())
        self.use_pretrained = word_embed_file is not None
        self.add_processor(
            'word', embed_file=word_embed_file, embed_size=word_embed_size,
            embed_dtype=embed_dtype,
            initializer=(lambda: np.random.normal(0, 1.0, word_embed_size)
                         .astype(embed_dtype, copy=False)),
            preprocess=word_preprocess, unknown=word_unknown)
        self.add_processor(
            'pos', embed_file=None, embed_size=pos_embed_size,
            embed_dtype=embed_dtype,
            initializer=(lambda: np.random.normal(0, 1.0, pos_embed_size)
                         .astype(embed_dtype, copy=False)),
            preprocess=lambda x: x.lower())
        self.label_map = text.Vocab()
        self._sentences = {}

    def map(self, item):
        # item -> (words, postags, (heads, labels))
        words = []
        postags = []
        heads = []
        labels = []
        for token in item:
            words.append(token['form'])
            postags.append(token['postag'])
            heads.append(token['head'])
            labels.append(self.label_map.add(token['deprel']))
        sample = (self._word_transform(words),
                  self.get_processor('pos').fit_transform(postags),
                  (np.array(heads, dtype=np.int32),
                   np.array(labels, dtype=np.int32)))
        sentence_id = ':'.join(str(word_id) for word_id in sample[0])
        self._sentences[sentence_id] = words
        self._count += 1
        sys.stderr.write("\rload samples: {}".format(self._count))
        sys.stderr.flush()
        return sample

    def load(self, file, train=False, size=None):
        self._count = 0
        if train and not self.use_pretrained:
            # assign an index if the given word is not in vocabulary
            self._word_transform = self.get_processor('word').fit_transform
        else:
            # return the unknown word index if the word is not in vocabulary
            self._word_transform = self.get_processor('word').transform
        return super(DataLoader, self).load(file, train, size)

    def get_sentence(self, word_ids, default=None):
        sentence_id = ':'.join(str(word_id) for word_id in word_ids)
        return self._sentences.get(sentence_id, default)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_sentences'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


TRAIN_FILE = '/Users/hiroki/Desktop/NLP/data/ptb-sd3.3.0/dep/wsj_02-21.conll'
TEST_FILE = '/Users/hiroki/Desktop/NLP/data/ptb-sd3.3.0/dep/wsj_22.conll'


class TestLoader(unittest.TestCase):

    def setUp(self):
        self.loader = DataLoader(word_embed_file=None,
                                 word_embed_size=100,
                                 pos_embed_size=50)

    def test_load(self):
        train_dataset = self.loader.load(TRAIN_FILE, train=True)
        test_dataset = self.loader.load(TEST_FILE, train=False)
        self.assertTrue(len(train_dataset) > 0)
        self.assertTrue(len(test_dataset) > 0)


if __name__ == "__main__":
    unittest.main()
