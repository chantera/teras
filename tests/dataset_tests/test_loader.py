import pathlib
import unittest

import numpy as np
from teras.dataset.loader import TextLoader
from teras.io.reader import ConllReader
from teras.preprocessing import text


class DataLoader(TextLoader):

    def __init__(self,
                 word_embed_size=100,
                 pos_embed_size=100,
                 word_embed_file=None,
                 word_preprocess=text.replace_number,
                 word_unknown="UNKNOWN",
                 embed_dtype='float32'):
        super().__init__(reader=ConllReader())
        self.use_pretrained = word_embed_file is not None
        word_vocab = text.EmbeddingVocab(
            word_unknown, file=word_embed_file, dim=word_embed_size,
            dtype=embed_dtype, initializer=text.EmbeddingVocab.random_normal)
        pos_vocab = text.EmbeddingVocab(
            dim=pos_embed_size, dtype=embed_dtype,
            initializer=text.EmbeddingVocab.random_normal)
        self.add_processor(
            'word', vocab=word_vocab, preprocess=word_preprocess)
        self.add_processor(
            'pos', vocab=pos_vocab, preprocess=lambda x: x.lower())
        self.label_map = text.Dict()

    def map(self, item):
        # item -> (words, postags, (heads, labels))
        words, postags, heads, labels = \
            zip(*[(token['form'], token['postag'], token['head'],
                   self.label_map.add(token['deprel'])) for token in item])
        sample = (self.map_attr('word', words,
                                self.train and not self.use_pretrained),
                  self.map_attr('pos', postags, True),
                  (np.array(heads, dtype=np.int32),
                   np.array(labels, dtype=np.int32)))
        return sample


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
        test_dataset = self.loader.load(TEST_FILE, train=False, size=100)
        self.assertTrue(len(train_dataset) > 0)
        self.assertTrue(len(test_dataset) > 0)


if __name__ == "__main__":
    unittest.main()
