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


if __name__ == "__main__":
    unittest.main()
