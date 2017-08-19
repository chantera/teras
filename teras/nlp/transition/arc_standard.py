from enum import IntEnum


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
        return not(state.end())

    @staticmethod
    def is_allowed_left_arc(state):
        return state.stack_size > 2

    @staticmethod
    def is_allowed_right_arc(state):
        return state.stack_size > 1

    @staticmethod
    def is_terminal(state):
        return state.end() and state.stack_size < 2

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
