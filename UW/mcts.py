import statistics
import itertools
import numpy as np
import random
import copy
from collections import defaultdict

class Nim():

    def __init__(self, n_rows: int = 4):
        self.initial_state = tuple([int((x + 1)) for x in range(0, n_rows * 2, 2)])
        self.possible_values_in_rows = []

        for idx in self.initial_state:
            temp_list = []
            for ldx in range(idx+1):
                temp_list.append(ldx)
            self.possible_values_in_rows.append(temp_list)

        self.states = (tuple(itertools.product(*self.possible_values_in_rows)))
        self.n_states = len(self.states)
        self.current_state = copy.deepcopy(self.initial_state)

        self.win_states = list()
        for idx in range(n_rows):
            list_of_zeros = [0] * n_rows
            list_of_zeros[idx] = 1
            self.win_states.append(tuple(list_of_zeros))
        self.win_states = tuple(self.win_states)

    def reset(self):
        self.current_state = self.initial_state
        return self.current_state

    def get_all_states(self):
        return self.states

    def is_terminal(self, state):
        if not any(state): return True
        return False

    def get_possible_actions(self, state):
        possible_actions = []

        if self.is_terminal(state):
            possible_actions.append(state)
            return tuple(possible_actions)

        for row_idx, number_in_row in enumerate(state):
            for number in range(1, number_in_row + 1):
                single_action = [0 for _ in range(len(state))]
                single_action[row_idx] = number
                single_action = tuple(single_action)
                possible_actions.append(single_action)

        return tuple(possible_actions)

    def step(self, action):
        self.current_state = tuple([idx_1 - idx_2 for idx_1, idx_2 in zip(self.current_state, action)])
        return self.current_state


class MonteCarloTreeSearchNode():
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0

        self.win_states = list()
        for idx in range(len(self.state)):
            list_of_zeros = [0] * len(self.state)
            list_of_zeros[idx] = 1
            self.win_states.append(tuple(list_of_zeros))
        self.win_states = tuple(self.win_states)
        self._untried_actions = self.untried_actions()

        # print('new')

    def untried_actions(self):
        self._untried_actions = list(self.get_legal_actions())
        return self._untried_actions

    def q(self):
        return self._results[1]

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.move(action)
        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.is_game_over()

    def rollout(self):

        current_rollout_state = self.state
        invert_reward = False
        while not self.is_game_over_state(current_rollout_state):
            possible_moves = self.get_legal_actions_state(current_rollout_state)
            action = self.rollout_policy(possible_moves)
            current_rollout_state = self.move_state(current_rollout_state, action)
            invert_reward = not invert_reward

        reward = self.game_result_state(current_rollout_state)
        return -1 * reward if invert_reward else reward

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
        choices_weights = [(child.q() / child.n()) + c_param * np.sqrt((2 * np.log(self.n()) / child.n())) for child in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):

        current_node = self

        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        simulation_no = 100

        for i in range(simulation_no):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        return self.best_child(c_param=0.1)

    def get_legal_actions(self):
        possible_actions = []

        if self.is_game_over():
            possible_actions.append(self.state)
            return tuple(possible_actions)

        for row_idx, number_in_row in enumerate(self.state):
            for number in range(1, number_in_row + 1):
                single_action = [0 for _ in range(len(self.state))]
                single_action[row_idx] = number
                single_action = tuple(single_action)
                possible_actions.append(single_action)

        return tuple(possible_actions)

    def get_legal_actions_state(self, state):
        possible_actions = []

        if self.is_game_over_state(state):
            possible_actions.append(state)
            return tuple(possible_actions)

        for row_idx, number_in_row in enumerate(state):
            for number in range(1, number_in_row + 1):
                single_action = [0 for _ in range(len(state))]
                single_action[row_idx] = number
                single_action = tuple(single_action)
                possible_actions.append(single_action)

        return tuple(possible_actions)

    def is_game_over(self):
        if not any(self.state): return True
        return False

    def is_game_over_state(self, state):
        if not any(state): return True
        return False


    def game_result(self):
        if not any(self.state):
            return -1

    def game_result_state(self, state):
        if not any(state):
            return -1

    def move(self, action):
        return tuple([idx_1 - idx_2 for idx_1, idx_2 in zip(self.state, action)])

    def move_state(self, state, action):
        return tuple([idx_1 - idx_2 for idx_1, idx_2 in zip(state, action)])


# root = MonteCarloTreeSearchNode(state=(1, 3, 5, 7))
# selected_node = root.best_action()
#
# print([child._results for child in root.children])
# print(root.q())


nim = Nim()
player_1_wins = 0
player_2_wins = 0

for _ in range(1000):
    nim.reset()
    turn = 0
    while not nim.is_terminal(nim.current_state):
        if not turn % 2:
            node = MonteCarloTreeSearchNode(state=nim.current_state)
            action = node.best_action().parent_action
            # print(node.q(), node.n())
            # print('\n')
            # print(nim.current_state, action)
            # print([child.parent_action for child in node.children])
            # print([child._results for child in node.children])
            # print(node.q())
        else:
            action = random.choice(nim.get_possible_actions(nim.current_state))

        nim.step(action)

        if nim.is_terminal(nim.current_state):
            # print('---------------------------------')
            if turn % 2:
                player_1_wins += 1
            else:
                player_2_wins += 1

        turn += 1

print(f'Algorithm winrate: {player_1_wins * 100 / (player_1_wins + player_2_wins)}%')
