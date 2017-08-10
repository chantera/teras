import unittest

from teras.io import reader


FILE1 = '/Users/hiroki/Desktop/NLP/data/treebank_3/data/wsj_00.mrg'
FILE2 = '/Users/hiroki/Desktop/NLP/data/Treebank3_coord/ext/00/wsj_0001.ext'  # perline  # NOQA


class TestTreeReader(unittest.TestCase):

    def setUp(self):
        self.reader = reader.TreeReader()

    def test_read(self):
        data = self.reader.read(FILE1)
        print(data[0])
        self.assertTrue(len(data) > 0)
        self.assertTrue(data[0][0][0] == 'S')
        data = self.reader.read(FILE2)
        print(data[0])
        self.assertTrue(len(data) > 0)
        self.assertTrue(data[0][0][0] == 'S')
        print(reader.read_tree(FILE1)[0])


if __name__ == "__main__":
    unittest.main()
