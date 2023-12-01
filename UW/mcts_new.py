from __future__ import division

import statistics
import itertools
import numpy as np
import random
import copy
from collections import defaultdict

import time
import math
import random

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

    def get_reward(self):

        reward = 0

        if self.current_state in self.win_states:
            reward = 1

        return reward

    def step(self, action):
        self.current_state = tuple([idx_1 - idx_2 for idx_1, idx_2 in zip(self.current_state, action)])
        return self.current_state


def randomPolicy(node):
    while not node.is_terminal():
        try:
            action = random.choice(node.get_possible_actions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(node))
        node = node.step(action)
    return node.get_reward()


class treeNode():
    def __init__(self, state, parent):

        self.initial_state = tuple([int((x + 1)) for x in range(0, 4 * 2, 2)])
        self.possible_values_in_rows = []
        for idx in self.initial_state:
            temp_list = []
            for ldx in range(idx + 1):
                temp_list.append(ldx)
            self.possible_values_in_rows.append(temp_list)

        self.states = (tuple(itertools.product(*self.possible_values_in_rows)))

        self.state = state
        self.isTerminal = self.is_terminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}
        self.current_player = 1

    def __str__(self):
        s=[]
        s.append("totalReward: %s"%(self.totalReward))
        s.append("numVisits: %d"%(self.numVisits))
        s.append("isTerminal: %s"%(self.isTerminal))
        s.append("possibleActions: %s"%(self.children.keys()))
        return "%s: {%s}"%(self.__class__.__name__, ', '.join(s))

    def getCurrentPlayer(self):
        return self.current_player

    def reset(self):
        self.state = self.initial_state
        return self.state

    def get_all_states(self):
        return self.states

    def is_terminal(self):
        if not any(self.state): return True
        return False

    def get_possible_actions(self):
        possible_actions = []

        if self.is_terminal():
            possible_actions.append(self.state)
            return tuple(possible_actions)

        for row_idx, number_in_row in enumerate(self.state):
            for number in range(1, number_in_row + 1):
                single_action = [0 for _ in range(len(self.state))]
                single_action[row_idx] = number
                single_action = tuple(single_action)
                possible_actions.append(single_action)

        return tuple(possible_actions)

    def get_reward(self):

        reward = 0

        if self.is_terminal():
            reward = -1

        return reward

    def step(self, action):
        new_state = tuple([idx_1 - idx_2 for idx_1, idx_2 in zip(self.state, action)])
        new_node = treeNode(new_state, self)
        if self.current_player == 1:
            new_node.current_player = -1
        elif self.current_player == -1:
            new_node.current_player = 1
        return new_node

class mcts():
    def __init__(self, iterationLimit=None, explorationConstant = (1 / math.sqrt(2)),
                 rolloutPolicy = randomPolicy):

        self.root = None
        self.searchLimit = iterationLimit
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState):

        self.root = treeNode(initialState, None)
        for i in range(self.searchLimit):
            self.executeRound()

        bestChild = self.getBestChild(self.root, 0.1)
        action=(action for action, node in self.root.children.items() if node is bestChild).__next__()

        return action

    def perform_simulation(self, initialState):
        self.root = treeNode(initialState, None)
        for i in range(self.searchLimit):
            self.executeRound()

    def get_best_action(self):
        bestChild = self.getBestChild(self.root, 0.1)
        action = ((action, node) for action, node in self.root.children.items() if node is bestChild).__next__()
        return action

    def executeRound(self):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        node = self.selectNode(self.root)
        reward = self.rollout(node)
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.get_possible_actions()
        for action in actions:
            if action not in node.children:
                newNode = node.step(action)
                # newNode = treeNode(new_state, node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = node.getCurrentPlayer() * child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

nim = Nim()
# play sarsa lambda vs random
player_1_wins = 0
player_2_wins = 0

# searcher = mcts(iterationLimit=1000)
# searcher.search(initialState=nim.current_state, current_player=1)
# next_action, next_node = searcher.get_best_action()
# print(next_node.current_player)
# nim.step(next_action)
# searcher.search(initialState=nim.current_state)
# next_action, next_node = searcher.get_best_action()
# print(next_node.current_player)
# print(next_action, next_node)
# searcher.root = next_node
# next_action, next_node = searcher.get_best_action()
# print(next_action, next_node)


for epoch in range(100):
    searcher = mcts(iterationLimit=100)
    nim.reset()
    turn = 0
    while not nim.is_terminal(nim.current_state):
        # print(nim.current_state)
        if not turn % 2:
            action = searcher.search(initialState=nim.current_state)
            next_action, next_node = searcher.get_best_action()
            print(next_node)
            # print(action)
            # searcher.root = node
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
    if not epoch % 10: print(epoch)

print(f'Algorithm winrate: {player_1_wins * 100 / (player_1_wins + player_2_wins)}%')