import random
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from .Direction import Direction
from .Position import Position
from .GameState import GameState

"""
A pacman is that yellow thing with a big mouth that can eat points and ghosts!
In this game, there can be more than one pacman and they can eat each other too.
"""


class Pacman(ABC):
    """
    Make your choice!
    You can make moves completely randomly if you want, the game won't allow you to make an invalid move.
    That's what invalid_move is for - it will be true if your previous choice was invalid.
    """

    @abstractmethod
    def make_move(self, game_state, invalid_move=False) -> Direction:
        pass

    """
    The game will call this once for each pacman at each time step.
    """

    @abstractmethod
    def give_points(self, points):
        pass

    @abstractmethod
    def on_win(self, result: Dict["Pacman", int]):
        pass

    """
    Do whatever you want with this info. The game will continue until all pacmans die or all points are eaten.
    """

    @abstractmethod
    def on_death(self):
        pass


class Pacman_244823(Pacman):
    def __init__(self, learning_mode: bool = True):  # , path_for_weights: str = None):
        self.name = 'Pacman_244823'
        self.n_weights = 20
        self.alpha = 0.002
        self.epsilon = 1
        self.d_epsilon = 0.00002
        self.min_epsilon = 0.01
        self.gamma = 0.9
        self.possible_actions = []
        self.learning_mode = learning_mode
        try:
            self.weights = np.loadtxt(fname='weights_244823.txt')
        except FileNotFoundError:
            self.weights = np.random.uniform(-1, 1, size=self.n_weights)

        self.info: Dict[str, Any] = {
            'state': None,
            'action': None,
            'reward': None,
            'next_state': None
        }

        self.current_score = 0
        self.scores = []
        self.directions = {
            Direction.UP: Position(0, -1),
            Direction.DOWN: Position(0, 1),
            Direction.LEFT: Position(-1, 0),
            Direction.RIGHT: Position(1, 0),
        }

        self.legal_actions = list(Direction)

    def feature_function(self, state, action) -> np.ndarray:

        def normalize(x, divisor): return (float(x) / divisor)

        features = []

        my_position = state.you['position']
        next_position = my_position + self.directions[action]

        eatable_pacmen = [pacman['position'] for pacman in state.other_pacmans if pacman['is_eatable']]
        non_eatable_pacmen = [pacman['position'] for pacman in state.other_pacmans if not pacman['is_eatable']]
        eatable_ghosts = [ghost['position'] for ghost in state.ghosts if ghost['is_eatable']]
        non_eatable_ghosts = [ghost['position'] for ghost in state.ghosts if not ghost['is_eatable']]

        TRESHOLD = [5, 10, 15]
        for idx in range(len(TRESHOLD)):
            significant_n_eatable_pacmen = 0
            for position in eatable_pacmen:
                if self.manhattan_dist(next_position, position) < TRESHOLD[idx]:
                    significant_n_eatable_pacmen += 1

            significant_n_non_eatable_pacmen = 0
            for position in non_eatable_pacmen:
                if self.manhattan_dist(next_position, position) < TRESHOLD[idx]:
                    significant_n_non_eatable_pacmen += 1

            significant_n_eatable_ghosts = 0
            for position in eatable_ghosts:
                if self.manhattan_dist(next_position, position) < TRESHOLD[idx]:
                    significant_n_eatable_ghosts += 1

            significant_n_non_eatable_ghosts = 0
            for position in non_eatable_ghosts:
                if self.manhattan_dist(next_position, position) < TRESHOLD[idx]:
                    significant_n_non_eatable_ghosts += 1

            significant_n_points = 0
            for point in state.points:
                if self.manhattan_dist(next_position, point) < TRESHOLD[idx]:
                    significant_n_points += 1

            features.append(normalize(significant_n_non_eatable_ghosts, TRESHOLD[idx] ** 2))
            features.append(normalize(significant_n_eatable_pacmen, TRESHOLD[idx] ** 2))
            features.append(normalize(significant_n_non_eatable_pacmen, TRESHOLD[idx] ** 2))
            features.append(normalize(significant_n_eatable_ghosts, TRESHOLD[idx] ** 2))
            features.append(normalize(significant_n_points, TRESHOLD[idx] ** 2))

        features.append(1) if next_position in state.points else features.append(0)
        features.append(1) if next_position in non_eatable_ghosts else features.append(0)
        features.append(1) if next_position in eatable_ghosts else features.append(0)
        features.append(1) if next_position in non_eatable_pacmen else features.append(0)
        features.append(1) if next_position in eatable_pacmen else features.append(0)

        return np.array(features)

    def function_value(self, state, action):
        return float(np.dot(self.weights, self.feature_function(state, action)))

    def get_best_action(self, state):
        possible_actions = self.legal_actions
        if len(possible_actions) == 0:
            return None

        possible_actions_dict = dict()
        for action in possible_actions:
            possible_actions_dict[action] = self.function_value(state, action)
        sorted_dict = sorted(possible_actions_dict.items(), key=lambda kv: kv[1])

        # try:
        return random.choice([k for k, v in possible_actions_dict.items() if v == sorted_dict[-1][-1]])
        # except IndexError:
        #     return random.choice(possible_actions)

    def update_self(self, state, action, reward, next_state, terminal=1):

        if None in self.info.values():
            return

        function_approx = self.function_value(next_state, self.get_best_action(next_state)) * terminal
        values = self.feature_function(next_state, action)
        error = (reward + self.gamma * function_approx - self.function_value(state, action))

        for idx in range(len(self.weights)):
            self.weights[idx] += self.alpha * error * values[idx]

        self.epsilon -= self.d_epsilon
        if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon
        # print(self.epsilon)

    def make_move(self, state: GameState, invalid_move: bool = False) -> Direction:
        if invalid_move:
            self.legal_actions.remove(self.info['action'])

        else:
            self.info['next_state'] = state

        if self.learning_mode:
            self.update_self(state=self.info['state'],
                             action=self.info['action'],
                             reward=self.info['reward'],
                             next_state=self.info['next_state'])
            if random.random() < self.epsilon:
                chosen_action = random.choice(self.legal_actions)
            else:
                chosen_action = self.get_best_action(state)


        else:
            chosen_action = self.get_best_action(state)

        if not invalid_move:
            self.info['state'] = state
            self.legal_actions = list(Direction)

        self.info['action'] = chosen_action
        return chosen_action

    def give_points(self, points):
        self.current_score += points
        self.info['reward'] = 10 * points

    def on_game_end(self):
        self.scores.append(self.current_score)
        self.current_score = 0
        # print(self.weights)
        self.save()

    def on_win(self, result):
        self.on_game_end()
        self.info['reward'] = 10
        self.update_self(state=self.info['state'],
                         action=self.info['action'],
                         reward=self.info['reward'],
                         next_state=self.info['next_state'], terminal=0)

    def on_death(self) -> None:
        self.on_game_end()
        self.info['reward'] = -100
        if self.learning_mode:
            self.update_self(state=self.info['state'],
                             action=self.info['action'],
                             reward=self.info['reward'],
                             next_state=self.info['next_state'], terminal=0)

    def save(self):
        np.savetxt(fname='weights_244823.txt', X=self.weights)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @staticmethod
    def manhattan_dist(start_point: Position, end_point: Position) -> float:
        return abs(start_point.x - end_point.x) + abs(start_point.y - end_point.y)


"""
I hope yours will be smarter than this one...
"""


class RandomPacman(Pacman):
    def __init__(self, print_status=True) -> None:
        self.print_status = print_status

    def give_points(self, points):
        if self.print_status:
            pass
            # print(f"random pacman got {points} points")

    def on_death(self):
        if self.print_status:
            pass
            # print("random pacman dead")

    def on_win(self, result: Dict["Pacman", int]):
        if self.print_status:
            pass
            # print("random pacman won")

    def make_move(self, game_state, invalid_move=False) -> Direction:
        return random.choice(list(Direction))  # it will make some valid move at some point
