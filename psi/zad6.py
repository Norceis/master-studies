import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()


def f(x, y): return (1.5 - x - x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2


RANGES = [-4.5, 4.5]
SWARM_COUNT = 100
EPOCHS = 50
WEIGHT = 0.85
CONFIDENCE = 0.5


def validate(swarmling, position):
    return RANGES[0] < position[0][swarmling] < RANGES[1] and RANGES[0] < position[1][swarmling] < RANGES[1]


def calculate_new_speed(swarmling, swarmling_best, speed, position):
    first = np.random.rand(2)
    second = np.random.rand(2)

    new_speed = WEIGHT * np.asarray((speed[0][swarmling], speed[1][swarmling])) \
                + CONFIDENCE * first * (np.asarray((swarmling_best[0][swarmling], swarmling_best[1][swarmling]))
                                        - np.asarray((position[0][swarmling], position[1][swarmling]))) \
                + CONFIDENCE * second * (swarm_best - np.asarray((position[0][swarmling], position[1][swarmling])))

    return new_speed


positions = []
swarm_bests = []
personal_bests = []

position = np.stack(
    (np.random.uniform(RANGES[0], RANGES[1], SWARM_COUNT), np.random.uniform(RANGES[0], RANGES[1], SWARM_COUNT)))
speed = np.stack((np.random.random(SWARM_COUNT), np.random.random(SWARM_COUNT)))

personal_best = np.copy(position)
personal_best_values = f(personal_best[0], personal_best[1])

minimum_value = np.argmin(personal_best_values)
swarm_best = np.asarray((personal_best[0][minimum_value], personal_best[1][minimum_value]))
swarm_best_value = f(swarm_best[0], swarm_best[1])

for _ in range(EPOCHS):
    positions.append(np.copy(position))
    swarm_bests.append((list(np.copy(swarm_best)), swarm_best_value))
    personal_bests.append(np.copy(personal_best))

    for swarmling in range(SWARM_COUNT):
        new_speed = calculate_new_speed(swarmling, personal_best, speed, position)
        speed[0][swarmling], speed[1][swarmling] = new_speed[0], new_speed[1]
        position[0][swarmling] += speed[0][swarmling]
        position[1][swarmling] += speed[1][swarmling]

        if validate(swarmling, position):
            personal_best_values[swarmling] = f(position[0][swarmling], position[1][swarmling])

            if swarm_best_value > personal_best_values[swarmling]:
                swarm_best = np.asarray((position[0][swarmling], position[1][swarmling]))
                swarm_best_value = personal_best_values[swarmling]

            if f(personal_best[0][swarmling], personal_best[1][swarmling]) > personal_best_values[swarmling]:
                personal_best[0][swarmling], personal_best[1][swarmling] = position[0][swarmling], position[1][
                    swarmling]
        else:
            if (position[0][swarmling] < RANGES[0] and speed[0][swarmling] < 0) or \
                    (position[0][swarmling] > RANGES[1] and speed[0][swarmling] > 0):
                speed[0][swarmling] *= -1
            if (position[1][swarmling] < RANGES[0] and speed[1][swarmling] < 0) or \
                    (position[1][swarmling] > RANGES[1] and speed[1][swarmling] > 0):
                speed[1][swarmling] *= -1


def animate(i):
    ax.clear()
    ax.set_xlim(RANGES[0], RANGES[1])
    ax.set_ylim(RANGES[0], RANGES[1])
    ax.set_title('Epoch #' + str(i) + '\nBest value: ' + str(round(swarm_bests[i][1], 5)) + '\nat x = '
                 + str(round(swarm_bests[i][0][0], 3)) + ', y = ' + str(round(swarm_bests[i][0][1], 3)))

    ax.scatter(swarm_bests[i][0][0], swarm_bests[i][0][1], c='magenta', s=125, alpha=0.25, label='Swarm best position')
    ax.scatter(personal_bests[i][0], personal_bests[i][1], c='teal', s=5, alpha=0.25,
               label='Swarmling best position')
    ax.scatter(positions[i][0], positions[i][1], c='blue', s=5, alpha=0.35, label='Swarmling')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)


to_gif = FuncAnimation(fig, animate, frames=EPOCHS, interval=150)
to_gif.save("swarm " + str(int(time.time())) + ".gif", dpi=500)
