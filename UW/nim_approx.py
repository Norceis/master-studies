import statistics
import itertools
import numpy as np
import random
import copy
from collections import defaultdict

from matplotlib import pyplot as plt


class ApproximatedAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):

        self.get_legal_actions = get_legal_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.weights = np.random.uniform(low=-1, high=1, size=2)

    def function_values(self, state):

        if state in nim.win_states:
            first_part = 1
            second_part = 0

        elif nim.is_terminal(state):
            first_part = 0
            second_part = 0

        else:
            first_part = 0
            if not int(nim_sum(state)):
                second_part = 1
            else:
                second_part = -0.6

        return first_part, second_part

    def get_qvalue(self, state, action):

        next_state = nim.get_next_states(state, action)
        first_part, second_part = self.function_values(next_state)

        return first_part * self.weights[0] + second_part * self.weights[1]

    def get_value(self, state):

        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return 0.0

        return max([self.get_qvalue(state, action) for action in possible_actions])

    def update(self, state, action, reward, next_state):

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        best_action_qvalue = self.get_qvalue(next_state, self.get_best_action(next_state))

        values = self.function_values(next_state)
        error = (reward + gamma * best_action_qvalue - self.get_qvalue(state, action))

        for idx in range(len(self.weights)):
            self.weights[idx] += learning_rate * error * values[idx]

    def get_best_action(self, state):

        possible_actions = self.get_legal_actions(state)

        if len(possible_actions) == 0:
            return None

        possible_actions_dict = dict()

        for action in possible_actions:
            possible_actions_dict[action] = self.get_qvalue(state, action)

        sorted_dict = sorted(possible_actions_dict.items(), key=lambda kv: kv[1])

        return random.choice([k for k, v in possible_actions_dict.items() if v == sorted_dict[-1][-1]])

    def get_action(self, state):

        possible_actions = self.get_legal_actions(state)

        if len(possible_actions) == 0:
            return None

        epsilon = self.epsilon

        if random.random() < epsilon:
            return random.choice(possible_actions)

        return self.get_best_action(state)

    def turn_off_learning(self):
        self.epsilon = 0
        self.alpha = 0


class Nim:

    def __init__(self, n_rows: int = 4):
        self.initial_state = tuple([int((x + 1)) for x in range(0, n_rows * 2, 2)])
        self.possible_values_in_rows = []
        for idx in self.initial_state:
            temp_list = []
            for ldx in range(idx + 1):
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
        next_state = tuple([idx_1 - idx_2 for idx_1, idx_2 in zip(state, action)])
        return next_state

    def get_number_of_states(self):
        return self.n_states

    def get_reward(self, state, action, next_state):
        assert action in self.get_possible_actions(
            state), "cannot do action %s from state %s" % (action, state)

        reward = 0

        # if self.is_terminal(next_state):
        #     reward += -1

        if next_state in self.win_states:
            reward += 1
        #
        # if not int(nim_sum(next_state)):
        #     reward += 0.15

        return reward

    def step(self, action):
        prev_state = self.current_state
        self.current_state = tuple([idx_1 - idx_2 for idx_1, idx_2 in zip(self.current_state, action)])
        return self.current_state, self.get_reward(prev_state, action, self.current_state), \
            self.is_terminal(self.current_state), None


def nim_sum(state):
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


def play_and_train_ql(env, agent, player=0):
    total_reward = 0.0
    state = env.reset()

    done = False
    turn = 0 if not player else 1

    # print('===================================================')
    while not done:
        if not turn % 2:
            # get agent to pick action given state state.
            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)
            # print(state, reward, next_state, agent.weights)
            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward
        else:
            action = random.choice(env.get_possible_actions(state))
            next_state, reward, done, _ = env.step(action)
            state = next_state
        turn += 1
        if done:
            break

    return total_reward


nim = Nim()

approximated_agent_first = ApproximatedAgent(alpha=0.02, epsilon=0.25, discount=0.99,
                                             get_legal_actions=nim.get_possible_actions)

approximated_agent_second = ApproximatedAgent(alpha=0.5, epsilon=0.25, discount=0.99,
                                              get_legal_actions=nim.get_possible_actions)
first_weight = []
second_weight = []
for i in range(10000):
    first_weight.append(approximated_agent_first.weights[0])
    second_weight.append(approximated_agent_first.weights[1])
    play_and_train_ql(nim, approximated_agent_first, 0)
    # play_and_train_ql(nim, approximated_agent_second, 1)

# play ql vs random
player_1_wins = 0
player_2_wins = 0

for _ in range(1000):
    nim.reset()
    turn = 0
    while not nim.is_terminal(nim.current_state):
        if not turn % 2:
            action = approximated_agent_first.get_best_action(nim.current_state)
        else:
            action = random.choice(nim.get_possible_actions(nim.current_state))

        nim.step(action)

        if nim.is_terminal(nim.current_state):
            if turn % 2:
                player_1_wins += 1
            else:
                player_2_wins += 1

        turn += 1

print(f'Algorithm on correct starting turn vs random winrate: {player_1_wins * 100 / (player_1_wins + player_2_wins)}%')

# player_1_wins = 0
# player_2_wins = 0
#
# for _ in range(10000):
#     nim.reset()
#     turn = 0
#     while not nim.is_terminal(nim.current_state):
#         if not turn % 2:
#             action = random.choice(nim.get_possible_actions(nim.current_state))
#         else:
#             # action = random.choice(nim.get_possible_actions(nim.current_state))
#             action = approximated_agent_first.get_best_action(nim.current_state)
#
#         nim.step(action)
#
#         if nim.is_terminal(nim.current_state):
#             if turn % 2:
#                 player_1_wins += 1
#             else:
#                 player_2_wins += 1
#
#         turn += 1
#
# print(f'Algorithm on incorrect starting turn vs random winrate: {player_1_wins * 100 / (player_1_wins + player_2_wins)}%')
# #
# player_1_wins = 0
# player_2_wins = 0
#
# for _ in range(10000):
#     nim.reset()
#     turn = 0
#     while not nim.is_terminal(nim.current_state):
#         if not turn % 2:
#             action = approximated_agent_first.get_best_action(nim.current_state)
#         else:
#             # action = random.choice(nim.get_possible_actions(nim.current_state))
#             action = approximated_agent_second.get_best_action(nim.current_state)
#
#         nim.step(action)
#
#         if nim.is_terminal(nim.current_state):
#             if turn % 2:
#                 player_1_wins += 1
#             else:
#                 player_2_wins += 1
#
#         turn += 1
#
# print(f'Algorithm starting on correct turns vs algorithm winrate: {player_1_wins * 100 / (player_1_wins + player_2_wins)}%')
# #
# player_1_wins = 0
# player_2_wins = 0
#
# for _ in range(10000):
#     nim.reset()
#     turn = 0
#     while not nim.is_terminal(nim.current_state):
#         if not turn % 2:
#             action = approximated_agent_second.get_best_action(nim.current_state)
#         else:
#             # action = random.choice(nim.get_possible_actions(nim.current_state))
#             action = approximated_agent_first.get_best_action(nim.current_state)
#
#         nim.step(action)
#
#         if nim.is_terminal(nim.current_state):
#             if turn % 2:
#                 player_1_wins += 1
#             else:
#                 player_2_wins += 1
#
#         turn += 1
#
# print(f'Algorithm starting on incorrect turns vs algorithm winrate: {player_1_wins * 100 / (player_1_wins + player_2_wins)}%')

plt.plot(first_weight, linewidth=0.5, label='is winning state')
plt.plot(second_weight, linewidth=0.5, label='nim sum=0')
plt.legend()
plt.ylabel('weight value')
plt.show()
