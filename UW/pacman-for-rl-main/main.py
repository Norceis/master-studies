from tqdm import tqdm

from pacman.Ghost import Ghosts
from pacman.Pacman import RandomPacman, Pacman_244823
from pacman.Game import Game
import numpy as np

board = ["*   g",
         "gwww ",
         " w*  ",
         " www ",
         "p + p"]

board_big = ["wwwwwwwwwwwwwwwwwwwwwwwwwwww",
             "wp***********ww***********pw",
             "w*wwww*wwwww*ww*wwwww*wwww*w",
             "w+wwww*wwwww*ww*wwwww*wwww+w",
             "w*wwww*wwwww*ww*wwwww*wwww*w",
             "w**************************w",
             "w*wwww*ww*wwwwwwww*ww*wwww*w",
             "w*wwww*ww*wwwwwwww*ww*wwww*w",
             "w*****iww****ww****wwd*****w",
             "wwwwww*wwwww ww wwwww*wwwwww",
             "wwwwww*wwwww ww wwwww*wwwwww",
             "wwwwww*ww          ww*wwwwww",
             "wwwwww*ww www  www ww*wwwwww",
             "wwwwww*ww wwwggwww ww*wwwwww",
             "   z  *   www  www   *  z   ",
             "wwwwww*ww wwwggwww ww*wwwwww",
             "wwwwww*ww wwwwwwww ww*wwwwww",
             "wwwwww*ww s      s ww*wwwwww",
             "wwwwww*ww wwwwwwww ww*wwwwww",
             "wwwwww*ww wwwwwwww ww*wwwwww",
             "w*****i******ww******d*****w",
             "w*wwww*wwwww*ww*wwwww*wwww*w",
             "w*wwww*wwwww*ww*wwwww*wwww*w",
             "w+**ww****************ww**+w",
             "www*ww*ww*wwwwwwww*ww*ww*www",
             "www*ww*ww*wwwwwwww*ww*ww*www",
             "w******ww****ww****ww******w",
             "w*wwwwwwwwww*ww*wwwwwwwwww*w",
             "w*wwwwwwwwww*ww*wwwwwwwwww*w",
             "wp************************pw",
             "wwwwwwwwwwwwwwwwwwwwwwwwwwww"]

agent = Pacman_244823(learning_mode=False)
while True:
    game = Game(board_big, [Ghosts.RED, Ghosts.PINK, Ghosts.BLUE, Ghosts.ORANGE],
                [agent, RandomPacman(), RandomPacman(), RandomPacman()], False, delay=0)
    print(f'Score is: {list(game.run().values())[0]}')
