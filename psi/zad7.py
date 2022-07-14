import numpy as np
import math

import skfuzzy
import skfuzzy as fuzz
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from skfuzzy import control as ctrl
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

d = 10  # range
a = 45  # angle
k = 1  # air
m = 1  # mass
g = 9.81 #gravity constant

def v_simple(d, a):
    return (d * g / np.sin(2 * math.radians(a))) ** (1/2)

def v_hard(d, a, k, m):
    return (d * k) / (m * np.sin(2 * math.radians(a)) * np.exp(-2 * np.sin(math.radians(a))))

def fuzzy_simple():

    range_values = np.arange(1,90,2)
    angle_values = np.arange(1,90,2)
    scores = pd.DataFrame(columns = ['Range', 'Angle', 'Fuzzy velocity value', 'Math velocity value', 'Difference'])

    distance = ctrl.Antecedent(np.arange(0, 101, 1), 'distance')
    angle = ctrl.Antecedent(np.arange(0, 91, 1), 'angle')
    velocity = ctrl.Consequent(np.arange(0, 80, 1), 'velocity')

    distance.automf(3)
    angle.automf(5)
    velocity.automf(5)
    # distance.view()
    # angle.view()
    # velocity.view()

    # poor mediocre average decent good
    rule1 = ctrl.Rule(distance['poor'] & angle['poor'], velocity['mediocre'])
    rule2 = ctrl.Rule(distance['poor'] & angle['mediocre'], velocity['poor'])
    rule3 = ctrl.Rule(distance['poor'] & angle['average'], velocity['poor'])
    rule4 = ctrl.Rule(distance['poor'] & angle['decent'], velocity['poor'])
    rule5 = ctrl.Rule(distance['poor'] & angle['good'], velocity['mediocre'])

    rule6 = ctrl.Rule(distance['average'] & angle['poor'], velocity['average'])
    rule7 = ctrl.Rule(distance['average'] & angle['mediocre'], velocity['mediocre'])
    rule8 = ctrl.Rule(distance['average'] & angle['average'], velocity['mediocre'])
    rule9 = ctrl.Rule(distance['average'] & angle['decent'], velocity['mediocre'])
    rule10 = ctrl.Rule(distance['average'] & angle['good'], velocity['average'])

    rule11 = ctrl.Rule(distance['good'] & angle['poor'], velocity['good'])
    rule12 = ctrl.Rule(distance['good'] & angle['mediocre'], velocity['good'])
    rule13 = ctrl.Rule(distance['good'] & angle['average'], velocity['decent'])
    rule14 = ctrl.Rule(distance['good'] & angle['decent'], velocity['good'])
    rule15 = ctrl.Rule(distance['good'] & angle['good'], velocity['good'])

    throwing_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])
    throwing = ctrl.ControlSystemSimulation(throwing_ctrl)

    for index_value in range(len(range_values)):
        throwing.input['distance'] = range_values[index_value]
        throwing.input['angle'] = angle_values[index_value]
        throwing.compute()
        fuzzy_velocity = throwing.output['velocity']
        math_velocity = v_simple(range_values[index_value], angle_values[index_value])

        scores.loc[len(scores.index)] = [range_values[index_value], angle_values[index_value], fuzzy_velocity,
                                         math_velocity, np.abs(fuzzy_velocity-math_velocity)]

    plt.plot(range(0, len(scores)), scores['Fuzzy velocity value'], color='red', label='fuzzy')
    plt.plot(range(0, len(scores)), scores['Math velocity value'], color='blue', label='math')
    plt.xlabel('Test sample')
    plt.ylabel('Initial velocity')
    plt.legend()
    plt.show()

    return scores

def fuzzy_hard():

    range_values = np.arange(1, 60, 5)
    angle_values = np.arange(1, 80, 6)
    resistance_values = np.arange(0, 1, 0.08)
    mass_values = np.arange(1, 4, 0.15)

    scores = pd.DataFrame(columns=['Range', 'Angle', 'Air resistance', 'Mass',
                                   'Fuzzy velocity value', 'Math velocity value', 'Difference'])

    distance = ctrl.Antecedent(np.arange(1, 86, 1), 'distance')
    angle = ctrl.Antecedent(np.arange(1, 86, 1), 'angle')
    resistance = ctrl.Antecedent(np.arange(0, 2, 0.05), 'resistance')
    mass = ctrl.Antecedent(np.arange(1, 8, 0.33), 'mass')
    velocity = ctrl.Consequent(np.arange(5, 80, 1), 'velocity')


    # distance.automf(3)
    # angle.automf(5)
    resistance.automf(3)
    # mass.automf(3)

    resistance['zero'] = skfuzzy.gaussmf(resistance.universe, 0, 0.2)
    # resistance['poor'] = skfuzzy.trimf(resistance.universe, [0, 0, 0.5])
    resistance['average'] = skfuzzy.trimf(resistance.universe, [0.3, 1, 1.3])
    resistance['good'] = skfuzzy.trimf(resistance.universe, [1, 1.5, 2])
    # resistance.view()

    mass['zero'] = skfuzzy.gaussmf(mass.universe, 0, 1)
    mass['poor'] = skfuzzy.trimf(mass.universe, [0.5, 1, 4])
    mass['average'] = skfuzzy.trimf(mass.universe, [2.5, 5, 6.5])
    mass['good'] = skfuzzy.trimf(mass.universe, [5, 6.5, 8])

    angle['zero'] = skfuzzy.gaussmf(angle.universe, 0, 10)
    angle['poor'] = skfuzzy.trimf(angle.universe, [0, 5, 20])
    angle['mediocre'] = skfuzzy.trimf(angle.universe, [10, 15, 30])
    angle['average'] = skfuzzy.trimf(angle.universe, [15, 30, 45])
    angle['decent'] = skfuzzy.trimf(angle.universe, [30, 45, 60])
    angle['good'] = skfuzzy.trimf(angle.universe, [45, 70, 90])

    distance['zero'] = skfuzzy.gaussmf(distance.universe, 0, 5)
    distance['poor'] = skfuzzy.trimf(distance.universe, [0, 10, 30])
    distance['average'] = skfuzzy.trimf(distance.universe, [20, 40, 50])
    distance['good'] = skfuzzy.trimf(distance.universe, [40, 65, 90])

    velocity['zero'] = skfuzzy.gaussmf(velocity.universe, 0, 5)
    velocity['unacceptable'] = skfuzzy.trimf(velocity.universe, [0, 10, 15])
    velocity['dismal'] = skfuzzy.trimf(velocity.universe, [5, 25, 40])
    velocity['poor'] = skfuzzy.trimf(velocity.universe, [25, 35, 50])
    velocity['mediocre'] = skfuzzy.trimf(velocity.universe, [30, 40, 50])
    velocity['average'] = skfuzzy.trimf(velocity.universe, [40, 50, 60])
    velocity['decent'] = skfuzzy.trimf(velocity.universe, [50, 55, 60])
    velocity['good'] = skfuzzy.trimf(velocity.universe, [55, 65, 80])
    velocity['excellent'] = skfuzzy.trimf(velocity.universe, [65, 70, 80])
    velocity['wonderful'] = skfuzzy.trimf(velocity.universe, [70, 80, 80])

    # distance.view()
    velocity.automf(9, names=['unacceptable', 'dismal', 'poor', 'mediocre', 'average', 'decent', 'good', 'excellent', 'wonderful'])

    # rules = (
    # ctrl.Rule(distance['poor'] & (angle['poor'] | angle['good']) & resistance['poor'] & mass['poor'], velocity['poor']),
    # ctrl.Rule(distance['poor'] & (angle['poor'] | angle['good']) & resistance['poor'] & mass['average'] | resistance['average'] & mass['poor'], velocity['mediocre']),
    # ctrl.Rule(distance['poor'] & (angle['poor'] | angle['good']) & resistance['average'] & mass['good'] | resistance['good'] & mass['average'], velocity['average']),
    # ctrl.Rule(distance['poor'] & (angle['poor'] | angle['good']) & resistance['average'] & mass['average'], velocity['mediocre'], velocity['mediocre']),
    # ctrl.Rule(distance['poor'] & (angle['poor'] | angle['good']) & resistance['good'] & mass['good'], velocity['decent'], velocity['average']),
    #
    # ctrl.Rule(distance['poor'] & (angle['mediocre'] | angle['decent']) & resistance['poor'] & mass['poor'], velocity['dismal']),
    # ctrl.Rule(distance['poor'] & (angle['mediocre'] | angle['decent']) & resistance['poor'] & mass['average'] | resistance['average'] & mass['poor'], velocity['poor']),
    # ctrl.Rule(distance['poor'] & (angle['mediocre'] | angle['decent']) & resistance['average'] & mass['good'] | resistance['good'] & mass['average'], velocity['mediocre']),
    # ctrl.Rule(distance['poor'] & (angle['mediocre'] | angle['decent']) & resistance['average'] & mass['average'], velocity['poor']),
    # ctrl.Rule(distance['poor'] & (angle['mediocre'] | angle['decent']) & resistance['good'] & mass['good'], velocity['mediocre']),
    #
    # ctrl.Rule(distance['poor'] & angle['average'] & resistance['poor'] & mass['poor'], velocity['dismal']),
    # ctrl.Rule(distance['poor'] & angle['average'] & resistance['poor'] & mass['average'] | resistance['average'] & mass['poor'], velocity['poor']),
    # ctrl.Rule(distance['poor'] & angle['average'] & resistance['average'] & mass['good'] | resistance['good'] & mass['average'], velocity['mediocre']),
    # ctrl.Rule(distance['poor'] & angle['average'] & resistance['average'] & mass['average'], velocity['poor']),
    # ctrl.Rule(distance['poor'] & angle['average'] & resistance['good'] & mass['good'], velocity['mediocre']),
    #
    # ctrl.Rule(distance['average'] & (angle['poor'] | angle['good']) & resistance['poor'] & mass['poor'], velocity['mediocre']),
    # ctrl.Rule(distance['average'] & (angle['poor'] | angle['good']) & resistance['poor'] & mass['average'] | resistance['average'] & mass['poor'], velocity['mediocre']),
    # ctrl.Rule(distance['average'] & (angle['poor'] | angle['good']) & resistance['average'] & mass['good'] | resistance['good'] & mass['average'], velocity['average']),
    # ctrl.Rule(distance['average'] & (angle['poor'] | angle['good']) & resistance['average'] & mass['average'], velocity['average']),
    # ctrl.Rule(distance['average'] & (angle['poor'] | angle['good']) & resistance['good'] & mass['good'], velocity['decent']),
    #
    # ctrl.Rule(distance['average'] & (angle['mediocre'] | angle['decent']) & resistance['poor'] & mass['poor'], velocity['mediocre']),
    # ctrl.Rule(distance['average'] & (angle['mediocre'] | angle['decent']) & resistance['poor'] & mass['average'] | resistance['average'] & mass['poor'], velocity['mediocre']),
    # ctrl.Rule(distance['average'] & (angle['mediocre'] | angle['decent']) & resistance['average'] & mass['good'] | resistance['good'] & mass['average'], velocity['average']),
    # ctrl.Rule(distance['average'] & (angle['mediocre'] | angle['decent']) & resistance['average'] & mass['average'], velocity['average']),
    # ctrl.Rule(distance['average'] & (angle['mediocre'] | angle['decent']) & resistance['good'] & mass['good'], velocity['decent']),
    #
    # ctrl.Rule(distance['average'] & angle['average'] & resistance['poor'] & mass['poor'], velocity['poor']),
    # ctrl.Rule(distance['average'] & angle['average'] & resistance['poor'] & mass['average'] | resistance['average'] & mass['poor'], velocity['poor']),
    # ctrl.Rule(distance['average'] & angle['average'] & resistance['average'] & mass['good'] | resistance['good'] & mass['average'], velocity['mediocre']),
    # ctrl.Rule(distance['average'] & angle['average'] & resistance['average'] & mass['average'], velocity['mediocre']),
    # ctrl.Rule(distance['average'] & angle['average'] & resistance['good'] & mass['good'], velocity['average']),
    #
    # ctrl.Rule(distance['good'] & (angle['poor'] | angle['good']) & resistance['poor'] & mass['poor'], velocity['average']),
    # ctrl.Rule(distance['good'] & (angle['poor'] | angle['good']) & resistance['poor'] & mass['average'] | resistance['average'] & mass['poor'], velocity['decent']),
    # ctrl.Rule(distance['good'] & (angle['poor'] | angle['good']) & resistance['average'] & mass['good'] | resistance['good'] & mass['average'], velocity['good']),
    # ctrl.Rule(distance['good'] & (angle['poor'] | angle['good']) & resistance['average'] & mass['average'], velocity['decent']),
    # ctrl.Rule(distance['good'] & (angle['poor'] | angle['good']) & resistance['good'] & mass['good'], velocity['good']),
    #
    # ctrl.Rule(distance['good'] & (angle['mediocre'] | angle['decent']) & resistance['poor'] & mass['poor'], velocity['mediocre']),
    # ctrl.Rule(distance['good'] & (angle['mediocre'] | angle['decent']) & resistance['poor'] & mass['average'] | resistance['average'] & mass['poor'], velocity['average']),
    # ctrl.Rule(distance['good'] & (angle['mediocre'] | angle['decent']) & resistance['average'] & mass['good'] | resistance['good'] & mass['average'], velocity['decent']),
    # ctrl.Rule(distance['good'] & (angle['mediocre'] | angle['decent']) & resistance['average'] & mass['average'], velocity['average']),
    # ctrl.Rule(distance['good'] & (angle['mediocre'] | angle['decent']) & resistance['good'] & mass['good'], velocity['decent']),
    #
    # ctrl.Rule(distance['good'] & angle['average'] & resistance['poor'] & mass['poor'], velocity['mediocre']),
    # ctrl.Rule(distance['good'] & angle['average'] & resistance['poor'] & mass['average'] | resistance['average'] & mass['poor'], velocity['average']),
    # ctrl.Rule(distance['good'] & angle['average'] & resistance['average'] & mass['good'] | resistance['good'] & mass['average'], velocity['decent']),
    # ctrl.Rule(distance['good'] & angle['average'] & resistance['average'] & mass['average'], velocity['average']),
    # ctrl.Rule(distance['good'] & angle['average'] & resistance['good'] & mass['good'], velocity['decent']),
    # )

    rules = (
        ctrl.Rule(distance['poor'] & resistance['poor'] & (angle['poor'] | angle['good']), velocity['poor']),
        ctrl.Rule(distance['poor'] & resistance['poor'] & (angle['mediocre'] | angle['decent']), velocity['dismal']),
        ctrl.Rule(distance['poor'] & resistance['poor'] & angle['average'], velocity['dismal']),
        ctrl.Rule(distance['poor'] & resistance['good'] & mass['poor'] & (angle['poor'] | angle['good']), velocity['mediocre']),
        ctrl.Rule(distance['poor'] & resistance['good'] & mass['poor'] & (angle['mediocre'] | angle['decent']), velocity['mediocre']),
        ctrl.Rule(distance['poor'] & resistance['good'] & mass['poor'] & angle['average'], velocity['mediocre']),
        ctrl.Rule(distance['poor'] & resistance['good'] & mass['good'] & (angle['poor'] | angle['good']), velocity['average']),
        ctrl.Rule(distance['poor'] & resistance['good'] & mass['good'] & (angle['mediocre'] | angle['decent']), velocity['poor']),
        ctrl.Rule(distance['poor'] & resistance['good'] & mass['good'] & angle['average'], velocity['dismal']),

        ctrl.Rule(distance['average'] & resistance['poor'] & (angle['poor'] | angle['good']), velocity['average']),
        ctrl.Rule(distance['average'] & resistance['poor'] & (angle['mediocre'] | angle['decent']), velocity['average']),
        ctrl.Rule(distance['average'] & resistance['poor'] & angle['average'], velocity['mediocre']),
        ctrl.Rule(distance['average'] & resistance['good'] & mass['poor'] & (angle['poor'] | angle['good']), velocity['excellent']),
        ctrl.Rule(distance['average'] & resistance['good'] & mass['poor'] & (angle['mediocre'] | angle['decent']), velocity['decent']),
        ctrl.Rule(distance['average'] & resistance['good'] & mass['poor'] & angle['average'], velocity['average']),
        ctrl.Rule(distance['average'] & resistance['good'] & mass['good'] & (angle['poor'] | angle['good']), velocity['good']),
        ctrl.Rule(distance['average'] & resistance['good'] & mass['good'] & (angle['mediocre'] | angle['decent']), velocity['average']),
        ctrl.Rule(distance['average'] & resistance['good'] & mass['good'] & angle['average'], velocity['mediocre']),

        ctrl.Rule(distance['good'] & resistance['poor'] & (angle['poor'] | angle['good']), velocity['excellent']),
        ctrl.Rule(distance['good'] & resistance['poor'] & (angle['mediocre'] | angle['decent']), velocity['decent']),
        ctrl.Rule(distance['good'] & resistance['poor'] & angle['average'], velocity['average']),
        ctrl.Rule(distance['good'] & resistance['good'] & mass['poor'] & (angle['poor'] | angle['good']), velocity['wonderful']),
        ctrl.Rule(distance['good'] & resistance['good'] & mass['poor'] & (angle['mediocre'] | angle['decent']), velocity['good']),
        ctrl.Rule(distance['good'] & resistance['good'] & mass['poor'] & angle['average'], velocity['decent']),
        ctrl.Rule(distance['good'] & resistance['good'] & mass['good'] & (angle['poor'] | angle['good']), velocity['excellent']),
        ctrl.Rule(distance['good'] & resistance['good'] & mass['good'] & (angle['mediocre'] | angle['decent']), velocity['decent']),
        ctrl.Rule(distance['good'] & resistance['good'] & mass['good'] & angle['average'], velocity['average'])
        )

    throwing_ctrl = ctrl.ControlSystem(rules)
    throwing = ctrl.ControlSystemSimulation(throwing_ctrl)

    for index_value in range(len(range_values)):
        throwing.input['distance'] = range_values[index_value]
        throwing.input['angle'] = angle_values[index_value]
        throwing.input['resistance'] = resistance_values[index_value]
        throwing.input['mass'] = mass_values[index_value]
        throwing.compute()
        fuzzy_velocity = throwing.output['velocity']
        math_velocity = v_hard(range_values[index_value], angle_values[index_value],
                               resistance_values[index_value], mass_values[index_value])

        scores.loc[len(scores.index)] = [range_values[index_value], angle_values[index_value],
                                         resistance_values[index_value], mass_values[index_value],
                                         fuzzy_velocity, math_velocity,
                                         np.abs(fuzzy_velocity - math_velocity)]

    plt.plot(range(0, len(scores)), scores['Fuzzy velocity value'], color='red', label='fuzzy')
    plt.plot(range(0, len(scores)), scores['Math velocity value'], color='blue', label='math')
    plt.xlabel('Test sample')
    plt.ylabel('Initial velocity')
    plt.legend()
    plt.show()

    return scores

    # velocity.view(sim=throwing)
    # plt.show()
    # print(throwing.output['velocity'])


fuzzy_simple()
# fuzzy_hard()
