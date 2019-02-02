import numpy as np
import pickle
import tempfile
import unittest

from teras.preprocessing import text


class TestVocab(unittest.TestCase):

    def setUp(self):
        pass

    def test_add(self):
        v = text.Vocab("<UNK>")
        self.assertEqual(len(v), 1)

        self.assertEqual(v.add("Pierre"), 1)
        self.assertEqual(v.add("Vinken"), 2)
        self.assertEqual(v["Pierre"], 1)
        self.assertEqual(v["Vinken"], 2)
        self.assertEqual(v.add("Pierre"), 1)
        self.assertEqual(v.add("Vinken"), 2)
        self.assertEqual(len(v), 3)

        self.assertFalse("nonexecutive" in v)
        self.assertFalse("director" in v)
        self.assertEqual(v["nonexecutive"], 0)
        self.assertEqual(v["director"], 0)
        self.assertEqual(v.add("nonexecutive"), 3)
        self.assertEqual(v.add("director"), 4)
        self.assertTrue("nonexecutive" in v)
        self.assertTrue("director" in v)
        self.assertEqual(v["nonexecutive"], 3)
        self.assertEqual(v["director"], 4)
        self.assertEqual(len(v), 5)

        self.assertEqual(v.unknown, 0)
        self.assertEqual(v.lookup(3), "nonexecutive")
        self.assertEqual(v.lookup(4), "director")
        self.assertRaises(KeyError, lambda: v.lookup(256))

    def test_from_words(self):
        words = ("Pierre Vinken , 61 years old , will join the board "
                 "as a nonexecutive director Nov. 29 .")
        v = text.Vocab.from_words(words.split(), "<UNK>")
        self.assertEqual(len(v), 18)
        self.assertEqual(v.add("will"), 7)
        self.assertEqual(v.lookup(14), "director")
        self.assertTrue("Nov." in v)
        self.assertEqual(v.unknown, 0)

    def test_embeddings(self):
        v = text.EmbeddingVocab(dim=32)
        self.assertEqual(v.add("Pierre"), 1)
        self.assertEqual(v.add("Vinken"), 2)
        self.assertEqual(v.get_embeddings().shape, (3, 32))
        self.assertEqual(v.add("nonexecutive"), 3)
        self.assertEqual(v.add("director"), 4)
        self.assertEqual(v.get_embeddings().shape, (5, 32))
        np.testing.assert_array_equal(v.get_embeddings(), v.get_embeddings())
        self.assertIs(v.get_embeddings(), v.get_embeddings())

        x1 = v.get_embeddings()
        self.assertEqual(v.add("61"), 5)
        self.assertEqual(v.add("years"), 6)
        self.assertEqual(v.add("old"), 7)
        x2 = v.get_embeddings()
        self.assertEqual(x2.shape, (8, 32))
        self.assertIsNot(x1, x2)
        np.testing.assert_array_equal(x1, x2[:5])

    def test_embeddings_from_words(self):
        words = ("Pierre Vinken , 61 years old , will join the board "
                 "as a nonexecutive director Nov. 29 .")
        v = text.EmbeddingVocab.from_words(words.split(), "<UNK>", dim=64)
        self.assertEqual(v.get_embeddings().shape, (18, 64))

    def test_embeddings_serialize_deserialize(self):
        v1 = text.EmbeddingVocab(serialize_embeddings=False)
        self.assertEqual(v1.add("Pierre"), 1)
        self.assertEqual(v1.add("Vinken"), 2)
        self.assertEqual(v1.get_embeddings().shape, (3, 50))
        with tempfile.TemporaryFile() as f:
            pickle.dump(v1, f)
            self.assertEqual(v1.get_embeddings().shape, (3, 50))
            f.seek(0)
            v2 = pickle.load(f)
            self.assertIsNot(v1, v2)
            self.assertEqual(v1._dict, v2._dict)
            self.assertRaises(RuntimeError, lambda: v2.get_embeddings())
            self.assertEqual(v2.add("nonexecutive"), 3)
            self.assertEqual(v2.add("director"), 4)
            self.assertRaises(RuntimeError, lambda: v2.get_embeddings())

        v3 = text.EmbeddingVocab(serialize_embeddings=True, dim=16)
        self.assertEqual(v3.add("Pierre"), 1)
        self.assertEqual(v3.add("Vinken"), 2)
        self.assertEqual(v3.get_embeddings().shape, (3, 16))
        with tempfile.TemporaryFile() as f:
            pickle.dump(v3, f)
            self.assertEqual(v3.get_embeddings().shape, (3, 16))
            f.seek(0)
            v4 = pickle.load(f)
            self.assertIsNot(v3, v4)
            self.assertEqual(v3._dict, v4._dict)
            self.assertEqual(v4.get_embeddings().shape, (3, 16))
            self.assertEqual(v4.add("nonexecutive"), 3)
            self.assertEqual(v4.add("director"), 4)
            self.assertEqual(v4.get_embeddings().shape, (5, 16))

    def test_embeddings_from_file(self):
        raw = ("Pierre -0.00066023 -0.6566 0.27843 -0.14767\n"
               "Vinken 0.10204 -0.12792 -0.8443 -0.12181\n"
               "61 0.24968 -0.41242 0.1217 0.34527\n"
               "years -0.19181 -1.8823 -0.76746 0.099051\n"
               "old -0.52287 -0.31681 0.00059213 0.0074449\n")
        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write(raw)
            f.flush()
            v1 = text.EmbeddingVocab(file=f.name, dtype=np.float64)
            self.assertEqual(len(v1), 6)
            x1 = v1.get_embeddings()
            self.assertEqual(x1.shape, (6, 4))
            np.testing.assert_array_equal(
                x1[1], [0.10204, -0.12792, -0.8443, -0.12181])


class TestPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = text.Preprocessor()

    def test_preprocessing(self):
        sentence = ("Pierre Vinken , 61 years old , will join the board "
                    "as a nonexecutive director Nov. 29 .")

        ids = self.preprocessor.transform(sentence)
        expected = np.zeros(18, dtype=np.int32)
        np.testing.assert_array_equal(ids, expected)

        ids = self.preprocessor.fit(sentence).transform(sentence)
        expected = np.array([1, 2, 3, 4, 5, 6, 3, 7, 8, 9, 10,
                             11, 12, 13, 14, 15, 16, 17], dtype=np.int32)
        np.testing.assert_array_equal(ids, expected)

        ids = self.preprocessor.fit_transform("The Lorillard spokeswoman")
        expected = np.array([9, 18, 19], dtype=np.int32)
        np.testing.assert_array_equal(ids, expected)

        ids = self.preprocessor.pad(ids, 8)
        expected = np.array([9, 18, 19, -1, -1, -1, -1, -1], dtype=np.int32)
        np.testing.assert_array_equal(ids, expected)

        ids = self.preprocessor.fit_transform(
            "the Dutch publishing group", length=12)
        expected = np.array(
            [9, 20, 21, 22, -1, -1, -1, -1, -1, -1, -1, -1], dtype=np.int32)
        np.testing.assert_array_equal(ids, expected)


if __name__ == "__main__":
    unittest.main()
