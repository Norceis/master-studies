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

    def get_next_states(self, state, action):
        assert action in self.get_possible_actions(
            state), "cannot do action %s from state %s" % (action, state)
        # return self.transition_probs[state][action]
        next_state = tuple([idx_1 - idx_2 for idx_1, idx_2 in zip(state, action)])
        return next_state

    def get_number_of_states(self):
        return self.n_states

    def nim_sum(self, state):
        binary_rows = [format(row, 'b') for row in state]
        max_len = max([len(row) for row in binary_rows])
        binary_rows = ['0' * (max_len - len(row)) + row for row in binary_rows]

        result = ''
        for idx in range(max_len):
            res = 0
            for row in binary_rows:
                res += int(row[idx])
                res %= 2
            result += str(res)
        return result

    def get_reward(self, state, action, next_state):
        assert action in self.get_possible_actions(
            state), "cannot do action %s from state %s" % (action, state)

        reward = -5

        if next_state in self.win_states:
            reward += 100

        elif not int(self.nim_sum(next_state)):
            reward += 10

        return reward

    def step(self, action):
        prev_state = self.current_state
        self.current_state = tuple([idx_1 - idx_2 for idx_1, idx_2 in zip(self.current_state, action)])
        return self.current_state, self.get_reward(prev_state, action, self.current_state), \
               self.is_terminal(self.current_state), None



def value_iteration(nim, gamma, theta):
    V = dict()

    for state in nim.get_all_states():
        V[state] = 0

    policy = dict()
    for current_state in nim.get_all_states():
        try:
            policy[current_state] = nim.get_possible_actions(current_state)[0]
        except IndexError:
            continue

    while True:
        last_mean_value = statistics.fmean(V.values())
        for current_state in nim.get_all_states():
            actions = nim.get_possible_actions(current_state)
            state_action_values = dict()

            for action in actions:

                state_action_values[action] = 0
                next_state = nim.get_next_states(current_state, action)
                state_action_values[action] += nim.get_reward(current_state, action, next_state) + gamma * V[next_state]

            V[current_state] = max(list(state_action_values.values()))
        if abs(statistics.fmean(V.values()) - last_mean_value) < theta:
            break

    for current_state in nim.get_all_states():

        state_action_values = dict()
        actions = nim.get_possible_actions(current_state)

        for action in actions:
            state_action_values[action] = 0
            next_state = nim.get_next_states(current_state, action)
            state_action_values[action] += (nim.get_reward(current_state, action, next_state) + gamma * V[next_state])
        # print(f'{current_state} -------- {state_action_values}')
        max_value_action = max(state_action_values, key=state_action_values.get)

        if policy[current_state] != max_value_action:
            policy[current_state] = max_value_action

    return policy, V

nim = Nim(4)
optimal_policy, optimal_value = value_iteration(nim, 0.9, 0.001)


# play (player 1) value iteration vs random
player_1_wins = 0
player_2_wins = 0

for _ in range(10000):
    nim.reset()
    turn = 0
    while not nim.is_terminal(nim.current_state):
        if not turn % 2:
            action = random.choice(nim.get_possible_actions(nim.current_state))
        else:
            action = optimal_policy[tuple(nim.current_state)]
        nim.step(action)
        if nim.is_terminal(nim.current_state):
            if turn % 2:
                player_1_wins += 1
            else:
                player_2_wins += 1
        turn += 1

print(f'Algorithm winrate: {player_2_wins * 100 / (player_1_wins + player_2_wins)}%')