
import pickle
import copy
import itertools
import random
import warnings

import numpy as np

warnings.filterwarnings('ignore')


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
        # return self.transition_probs[state][action]
        next_state = tuple([idx_1 - idx_2 for idx_1, idx_2 in zip(state, action)])
        return next_state

    def get_number_of_states(self):
        return self.n_states

    def get_reward(self, state, action, next_state):
        assert action in self.get_possible_actions(
            state), "cannot do action %s from state %s" % (action, state)

        reward = 0

        if next_state in self.win_states:
            reward += 10

        elif self.is_terminal(next_state):
            reward += -10

        return reward

    def step(self, action):
        prev_state = self.current_state
        self.current_state = tuple([idx_1 - idx_2 for idx_1, idx_2 in zip(self.current_state, action)])
        return self.current_state, self.get_reward(prev_state, action, self.current_state), \
            self.is_terminal(self.current_state), None


class WAN(object):  #Weight Agnostic Neural
    def __init__(self, init_shared_weight):
        self.input_size = 4
        self.aVec = np.random.randint(0, 10, 20)
        self.wKey = np.random.randint(1, 400, 50)
        self.weights = np.random.normal(0, 1, 50)
        self.weight_bias = -1.5
        nNodes = len(self.aVec)
        self.wVec = [0] * (nNodes * nNodes)
        self.set_weight(init_shared_weight, 0)

    def set_weight(self, weight, weight_bias):
        nValues = len(self.wKey)
        if isinstance(weight, (list, np.ndarray)):
            weights = weight
        else:
            weights = [weight] * nValues

        for i in range(nValues):
            k = self.wKey[i]
            self.wVec[k] = weights[i] + weight_bias

    def tune_weights(self):
        self.set_weight(self.weights, self.weight_bias)

    def mutate(self, winrate):
        mut_chance = np.exp(-6 * winrate + 3)

        for gate_idx in range(len(self.aVec)):
            if np.random.rand() < mut_chance * 0.05:
                new_gate = np.random.randint(0, 10)
                while self.aVec[gate_idx] == new_gate:
                    new_gate = np.random.randint(0, 10)
                self.aVec[gate_idx] = new_gate

        for connection_idx in range(len(self.wKey)):
            if np.random.rand() < mut_chance * 0.01:
                new_connection = np.random.randint(1, 400)
                self.wKey[connection_idx] = new_connection

    def get_possible_actions(self, state):
        possible_actions = []

        if state == (0, 0, 0, 0):
            possible_actions.append(state)
            return tuple(possible_actions)

        for row_idx, number_in_row in enumerate(state):
            for number in range(1, number_in_row + 1):
                single_action = [0 for _ in range(len(state))]
                single_action[row_idx] = number
                single_action = tuple(single_action)
                possible_actions.append(single_action)

        return tuple(possible_actions)

    def get_action_by_idx(self, old_state, indices):
        move_dict = {0: (1, 0, 0, 0),
                     1: (0, 1, 0, 0),
                     2: (0, 2, 0, 0),
                     3: (0, 3, 0, 0),
                     4: (0, 0, 1, 0),
                     5: (0, 0, 2, 0),
                     6: (0, 0, 3, 0),
                     7: (0, 0, 4, 0),
                     8: (0, 0, 5, 0),
                     9: (0, 0, 0, 1),
                     10: (0, 0, 0, 2),
                     11: (0, 0, 0, 3),
                     12: (0, 0, 0, 4),
                     13: (0, 0, 0, 5),
                     14: (0, 0, 0, 6),
                     15: (0, 0, 0, 7)}

        possible_actions = self.get_possible_actions(old_state)
        if (0, 0, 0, 0) == possible_actions[0]:
            return tuple[0, 0, 0, 0]

        for idx in indices:
            if move_dict[idx] in possible_actions:
                return move_dict[idx]

    def get_action(self, old_state):

        nNodes = len(self.aVec)
        wMat = np.array(self.wVec).reshape((nNodes, nNodes))
        nodeAct = [0] * nNodes

        for i in range(len(old_state)):
            nodeAct[i] = old_state[i]

        for iNode in range(self.input_size, nNodes):
            rawAct = np.dot(nodeAct, wMat[:, iNode:iNode + 1])
            rawAct = self.activate(self.aVec[iNode], rawAct.tolist()[0])
            nodeAct[iNode] = rawAct

        sorted_indices = np.argsort(-1 * np.array(nodeAct[-16:]))
        action = self.get_action_by_idx(old_state, sorted_indices)
        return action

    def activate(self, gate_idx, x):
        if gate_idx == 1:
            return x
        elif gate_idx == 2:
            return np.where(x >= 0, 1, 0)
        elif gate_idx == 3:
            return np.sin(np.pi * x)
        elif gate_idx == 4:
            return np.exp(-(x * x) / 2.0)
        elif gate_idx == 5:
            return np.tanh(x)
        elif gate_idx == 6:
            return 1.0 / (1.0 + np.exp(-x))
        elif gate_idx == 7:
            return -x
        elif gate_idx == 8:
            return np.abs(x)
        elif gate_idx == 9:
            return np.max(x, 0)
        elif gate_idx == 0:
            return np.cos(np.pi * x)
        else:
            return None


def wan(environment):
    # drl = WAN(-1.5)
    with open('0.783_test.pickle', 'rb') as handle:
        drl = pickle.load(handle)

    best_winrate = 0.78
    while True:
        # Training loop
        wan_wins = 0
        epochs = 50
        while True:
            for epoch in range(epochs):
                state_old = environment.reset()
                turn = 0
                while True:
                    if not turn % 2:
                        action_now = drl.get_action(state_old)
                        state_new, reward_now, done, _ = environment.step(action_now)
                        if done:
                            break
                        state_old = state_new
                        turn += 1
                    else:
                        action_now = random.choice((drl.get_possible_actions(state_old)))
                        state_new, reward_now, done, _ = environment.step(action_now)
                        if done:
                            wan_wins += 1
                            break
                        state_old = state_new
                        turn += 1
            if wan_wins / epochs > 0.8:
                # with open(str(wan_wins/epochs) + '_training.pickle', 'wb') as handle:
                #     pickle.dump(drl, handle)
                print(f'Winrate of WAN model in training is: {wan_wins / epochs * 100}')
                break
            else:
                drl.mutate(wan_wins / epochs)
                drl.tune_weights()
                wan_wins = 0

        # Test loop
        wan_wins = 0
        epochs = 1000
        for epoch in range(epochs):
            state_old = environment.reset()
            turn = 0
            while True:
                if not turn % 2:
                    action_now = drl.get_action(state_old)
                    state_new, reward_now, done, _ = environment.step(action_now)
                    if done:
                        break
                    state_old = state_new
                    turn += 1
                else:
                    action_now = random.choice((drl.get_possible_actions(state_old)))
                    state_new, reward_now, done, _ = environment.step(action_now)
                    if done:
                        wan_wins += 1
                        break
                    state_old = state_new
                    turn += 1

        print(f'Winrate of WAN model in testing is: {wan_wins / epochs * 100}')
        print(f'')
        if wan_wins / epochs > best_winrate:
            with open(str(wan_wins/epochs) + '_test.pickle', 'wb') as handle:
                pickle.dump(drl, handle)
            # break
        else:
            drl.mutate(wan_wins / epochs)
            drl.tune_weights()

wan(Nim())

with open('0.791_test.pickle', 'rb') as handle:
    drl = pickle.load(handle)
nim = Nim()
wan_wins = 0
epochs = 10000
for epoch in range(epochs):
    state_old = nim.reset()
    turn = 0
    while True:
        if not turn % 2:
            action_now = drl.get_action(state_old)
            state_new, reward_now, done, _ = nim.step(action_now)
            if done:
                break
            state_old = state_new
            turn += 1
        else:
            action_now = random.choice((drl.get_possible_actions(state_old)))
            state_new, reward_now, done, _ = nim.step(action_now)
            if done:
                wan_wins += 1
                break
            state_old = state_new
            turn += 1

print(f'Winrate of WAN model in testing is: {wan_wins / epochs * 100}')