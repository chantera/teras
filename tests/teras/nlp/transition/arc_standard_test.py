import sys
from types import SimpleNamespace
import unittest

from teras.io.reader import ConllReader
from teras.nlp.transition import arc_standard
from teras.nlp.transition.state import State
from teras.preprocessing import text


class Token(SimpleNamespace):
    pass


class ConllLoader(object):

    def __init__(self):
        self.reader = ConllReader()
        self.word_processor = text.Preprocessor()
        self.pos_processor = text.Preprocessor()
        self.label_map = text.Vocab()

    def load(self, file):
        self.reader.set_file(file)
        sentences = [[Token(
            id=token['id'],
            form=self.word_processor.fit_transform(token['form'])[0],
            postag=self.pos_processor.fit_transform(token['postag'])[0],
            head=token['head'],
            label=self.label_map.add(token['deprel'])
        ) for token in tokens] for tokens in self.reader]
        return sentences


CONLL_FILE = '/Users/hiroki/Desktop/NLP/data/ptb-sd3.3.0/dep/wsj_02-21.conll'

_DATA = [None]


def load_data(reload=False):
    if _DATA[0] is None or reload:
        _DATA[0] = ConllLoader().load(CONLL_FILE)
        print("data size: {} sentences".format(len(_DATA[0])))
    return _DATA[0]


class TestArcStandard(unittest.TestCase):

    def setUp(self):
        self.data = load_data()

    def test_shift(self):
        state = State(self.data[0])
        self.assertEqual(state.stack_top, 0)
        self.assertEqual(state.buffer_head, 1)
        arc_standard.ArcStandard.shift(state)
        self.assertEqual(state.stack_top, 1)
        self.assertEqual(state.buffer_head, 2)
        self.assertEqual(state.stack_size, 2)

    def test_left_arc(self):
        state = State(self.data[0])
        self.assertFalse(arc_standard.ArcStandard.is_allowed_left_arc(state))
        arc_standard.ArcStandard.shift(state)
        self.assertFalse(arc_standard.ArcStandard.is_allowed_left_arc(state))
        arc_standard.ArcStandard.shift(state)
        self.assertTrue(arc_standard.ArcStandard.is_allowed_left_arc(state))
        arc_standard.ArcStandard.shift(state)  # s[0,1,2,3] b[4,...
        self.assertTrue(arc_standard.ArcStandard.is_allowed_left_arc(state))
        self.assertEqual(state.stack_top, 3)
        self.assertEqual(state.buffer_head, 4)
        self.assertEqual(state.stack_size, 4)

        action = arc_standard.ArcStandard.left_arc_action(3)
        self.assertEqual(action, 7)
        arc_standard.ArcStandard.apply(action, state)  # s[0,1,3] b[4,...
        self.assertEqual(state.stack_top, 3)
        self.assertEqual(state.stack(1), 1)
        self.assertEqual(state.stack_size, 3)
        self.assertEqual(state.head(2), 3)
        self.assertEqual(state.label(2), 3)

        action = arc_standard.ArcStandard.left_arc_action(2)
        self.assertEqual(action, 5)
        arc_standard.ArcStandard.apply(action, state)  # s[0,3] b[4,...
        self.assertEqual(state.stack_top, 3)
        self.assertEqual(state.stack(1), 0)
        self.assertEqual(state.stack_size, 2)
        self.assertEqual(state.head(1), 3)
        self.assertEqual(state.label(1), 2)

        self.assertFalse(arc_standard.ArcStandard.is_allowed_left_arc(state))

    def test_right_arc(self):
        state = State(self.data[0])
        self.assertFalse(arc_standard.ArcStandard.is_allowed_right_arc(state))
        arc_standard.ArcStandard.shift(state)
        self.assertTrue(arc_standard.ArcStandard.is_allowed_right_arc(state))
        arc_standard.ArcStandard.shift(state)
        self.assertTrue(arc_standard.ArcStandard.is_allowed_right_arc(state))
        arc_standard.ArcStandard.shift(state)  # s[0,1,2,3] b[4,...
        self.assertTrue(arc_standard.ArcStandard.is_allowed_right_arc(state))
        self.assertEqual(state.stack_top, 3)
        self.assertEqual(state.buffer_head, 4)
        self.assertEqual(state.stack_size, 4)

        action = arc_standard.ArcStandard.right_arc_action(3)
        self.assertEqual(action, 8)
        arc_standard.ArcStandard.apply(action, state)  # s[0,1,2] b[4,...
        self.assertEqual(state.stack_top, 2)
        self.assertEqual(state.stack(1), 1)
        self.assertEqual(state.stack_size, 3)
        self.assertEqual(state.head(3), 2)
        self.assertEqual(state.label(3), 3)

        action = arc_standard.ArcStandard.right_arc_action(2)
        self.assertEqual(action, 6)
        arc_standard.ArcStandard.apply(action, state)  # s[0,1] b[4,...
        self.assertEqual(state.stack_top, 1)
        self.assertEqual(state.stack(1), 0)
        self.assertEqual(state.stack_size, 2)
        self.assertEqual(state.head(2), 1)
        self.assertEqual(state.label(2), 2)

        self.assertTrue(arc_standard.ArcStandard.is_allowed_right_arc(state))

    # def test_oracle_one(self):
    #     state = State(self.data[0])
    #     actions = []
    #     while not arc_standard.ArcStandard.is_terminal(state):
    #         action = arc_standard.ArcStandard.get_oracle(state)
    #         actions.append(action)
    #         arc_standard.ArcStandard.apply(action, state)
    #     self.assertEqual(actions, [0, 0, 3, 0, 8, 0, 0, 9, 0, 11, 14, 0, 8, 0, 0, 15, 5, 0, 0, 17, 20, 0, 0, 0, 0, 13, 17, 24, 22, 0, 0, 10, 26, 0, 8, 2])  # for the first sample of wsj_00  # NOQA
    #     self.assertEqual(state.stack_size, 1)
    #     self.assertEqual(state.stack_top, 0)

    def test_oracle(self):
        for i, tokens in enumerate(self.data):
            golds = [token.head for token in tokens]
            arc_standard.projectivize(golds)
            for token, gold in zip(tokens, golds):
                token.head = gold
            state = State(tokens)
            while not arc_standard.ArcStandard.is_terminal(state):
                action = arc_standard.ArcStandard.get_oracle(state)
                arc_standard.ArcStandard.apply(action, state)
            correct = 0
            for index in range(state.num_tokens):
                if state.head(index) == state.token(index).head and \
                        state.label(index) == state.token(index).label:
                    correct += 1
            sys.stderr.write("\rassertion: {}".format(i))
            self.assertTrue(correct == state.num_tokens)


if __name__ == "__main__":
    unittest.main()
