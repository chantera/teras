import logging
import tempfile
import unittest

from teras.io import cache
from teras.preprocessing import text


SENTENCE = ("Pierre Vinken , 61 years old , will join the board "
            "as a nonexecutive director Nov. 29 .")


class TestCache(unittest.TestCase):

    def setUp(self):
        self.vocab = text.Vocab.from_words(SENTENCE.split())

    def test_dump_load(self):
        with tempfile.TemporaryDirectory() as d:
            c = cache.Cache(key=SENTENCE, dir=d, prefix='vocab-')
            self.assertRaises(FileNotFoundError, lambda: c.load())
            c.dump(self.vocab)
            v = c.load()
            self.assertEqual(len(v), 18)
            self.assertEqual(v.add("will"), 7)
            self.assertEqual(v.lookup(14), "director")
            self.assertTrue("Nov." in v)
            self.assertEqual(v.unknown, 0)
            self.assertEqual(v._dict, self.vocab._dict)

    def test_load_or_create(self):
        logging.basicConfig()
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        with tempfile.TemporaryDirectory() as d:
            c = cache.Cache(key=SENTENCE, dir=d, prefix='vocab-',
                            logger=logger)
            self.assertRaises(FileNotFoundError, lambda: c.load())
            v1 = c.load_or_create(lambda: self.vocab)
            self.assertEqual(v1._dict, self.vocab._dict)
            v2 = c.load()
            self.assertEqual(v2._dict, self.vocab._dict)
            self.assertIsNot(v1._dict, v2._dict)

    def test_dump_load_func(self):
        with tempfile.TemporaryDirectory() as d:
            cache.dump(self.vocab, (SENTENCE, 'key2'), dir=d, prefix='vocab-')
            v = cache.load(key=(SENTENCE, 'key2'), dir=d, prefix='vocab-')
            self.assertEqual(v._dict, self.vocab._dict)
        with tempfile.TemporaryDirectory() as d:
            v = cache.load_or_create(
                key=(SENTENCE, 'key3'), factory=lambda: self.vocab,
                dir=d, prefix='vocab-')
            self.assertEqual(v._dict, self.vocab._dict)


if __name__ == "__main__":
    unittest.main()
