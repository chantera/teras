from enum import IntEnum


def projectivize(heads):
    """https://github.com/tensorflow/models/blob/7d30a017fe50b648be6dee544f8059bde52db562/syntaxnet/syntaxnet/document_filters.cc#L296"""  # NOQA
    num_tokens = len(heads)
    while True:
        left = [-1] * num_tokens
        right = [num_tokens] * num_tokens

        for i, head in enumerate(heads):
            l = min(i, head)
            r = max(i, head)
            for j in range(l + 1, r):
                if left[j] < l:
                    left[j] = l
                if right[j] > r:
                    right[j] = r

        deepest_arc = -1
        max_depth = 0
        for i, head in enumerate(heads):
            if head == 0:
                continue
            l = min(i, head)
            r = max(i, head)
            left_bound = max(left[l], left[r])
            right_bound = min(right[l], right[r])

            if l < left_bound or r > right_bound:
                depth = 0
                j = i
                while j != 0:
                    depth += 1
                    j = heads[j]
                if depth > max_depth:
                    deepest_arc = i
                    max_depth = depth

        if deepest_arc == -1:
            return True

        lifted_head = heads[heads[deepest_arc]]
        heads[deepest_arc] = lifted_head


class ArcStandard(object):

    class ActionType(IntEnum):
        SHIFT = 0
        LEFT_ARC = 1
        RIGHT_ARC = 2

    @staticmethod
    def num_action_types():
        pass

    @staticmethod
    def num_actions():
        pass

    @staticmethod
    def shift_action():
        return ArcStandard.ActionType.SHIFT

    @staticmethod
    def left_arc_action(label):
        return ArcStandard.ActionType.LEFT_ARC + (label << 1)

    @staticmethod
    def right_arc_action(label):
        return ArcStandard.ActionType.RIGHT_ARC + (label << 1)

    @staticmethod
    def action_type(action):
        return ArcStandard.ActionType(
            action if action < 1 else 1 + (~action & 1))

    @staticmethod
    def label(action):
        return -1 if action < 1 else (action - 1) >> 1

    @staticmethod
    def apply(action, state):
        action_type = ArcStandard.action_type(action)
        if action_type == ArcStandard.ActionType.SHIFT:
            ArcStandard.shift(state)
        elif action_type == ArcStandard.ActionType.LEFT_ARC:
            ArcStandard.left_arc(state, ArcStandard.label(action))
        elif action_type == ArcStandard.ActionType.RIGHT_ARC:
            ArcStandard.right_arc(state, ArcStandard.label(action))
        else:
            raise
        state.record(action)

    """Shift: (s, i|b, A) => (s|i, b, A)"""
    @staticmethod
    def shift(state):
        state.push(state.buffer_head)
        state.advance()

    """Left Arc: (s|i|j, b, A) => (s|j, b, A +(j,l,i))"""
    @staticmethod
    def left_arc(state, label):
        s0 = state.pop()
        s1 = state.pop()
        state.add_arc(s1, s0, label)
        state.push(s0)

    """Right Arc: (s|i|j, b, A) => (s|i, b, A +(i,l,j))"""
    @staticmethod
    def right_arc(state, label):
        s0 = state.pop()
        s1 = state.stack_top
        state.add_arc(s0, s1, label)

    @staticmethod
    def is_allowed(action, state):
        action_type = ArcStandard.action_type(action)
        if action_type == ArcStandard.ActionType.SHIFT:
            return ArcStandard.is_allowed_shift(state)
        elif action_type == ArcStandard.ActionType.LEFT_ARC:
            return ArcStandard.is_allowed_left_arc(state)
        elif action_type == ArcStandard.ActionType.RIGHT_ARC:
            return ArcStandard.is_allowed_right_arc(state)
        return False

    @staticmethod
    def is_allowed_shift(state):
        return not(state.buffer_empty())

    @staticmethod
    def is_allowed_left_arc(state):
        return state.stack_size > 2

    @staticmethod
    def is_allowed_right_arc(state):
        return state.stack_size > 1

    @staticmethod
    def is_terminal(state):
        return state.buffer_empty() and state.stack_size < 2

    @staticmethod
    def get_oracle(state):
        if state.stack_size < 2:
            return ArcStandard.shift_action()
        s0 = state.token(state.stack(0))
        s1 = state.token(state.stack(1))
        if s0.head == s1.id and \
                ArcStandard.done_right_children_of(state, s0.id):
            return ArcStandard.right_arc_action(s0.label)
        if s1.head == s0.id:
            return ArcStandard.left_arc_action(s1.label)
        return ArcStandard.shift_action()

    @staticmethod
    def done_right_children_of(state, head):
        index = state.buffer_head
        while index < state.num_tokens:
            actual_head = state.token(index).head
            if actual_head == head:
                return False
            index = head if head > index else index + 1
        return True
