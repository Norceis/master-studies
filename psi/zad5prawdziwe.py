import time
import pandas as pd
import numpy as np
import random as rand
import copy

df = pd.read_excel('GA_task.xlsx')
one_setup = []

for i in range(int(len(df.columns) / 2)):
    one_setup.append([i])
    for j in range(len(df)):
        one_setup[i].append([df.iloc[j, (2 * i)], df.iloc[j, (2 * i + 1)]])

sorted_setup = sorted(one_setup, key=lambda x: x[1][1], reverse=True)

dictionary_of_setup = {}
for process in sorted_setup:
    key, value = process[0], process[1:]
    dictionary_of_setup[key] = value
sorted_setup = dictionary_of_setup


def complete_simulation(init_pop, population, mut_chance, num_mut_places,
                        cross_chance, num_cross_places, selection_surv_percent, epochs):
    class FactoryQueue:

        def __init__(self, genotype: dict, fenotype: int, list_of_processes: list):
            self.genotype = genotype
            self.fenotype = fenotype
            self.list_of_processes = list_of_processes

        def __str__(self):
            return "Factory Queue, order of processes: " + str(self.genotype) + "\nTheir cost: " + str(self.fenotype)

        def __repr__(self):
            return "Factory Queue, order of processes: " + str(self.genotype) + ", \n their cost: " + str(self.fenotype)

        def __add__(self, other):
            split_places = []
            list_of_processes_A = []
            list_of_processes_B = []
            genotype_for_A = {}
            genotype_for_B = {}

            for i in range(num_cross_places):
                split_places.append(rand.randint(1, len(self.list_of_processes) - 2))
            split_places.sort()

            for i in range(num_cross_places):
                list_of_processes_A += (self.list_of_processes[:split_places[i]])
                list_of_processes_A += [x for x in other.list_of_processes[split_places[i]:] if x not in list_of_processes_A]
                list_of_processes_B += (other.list_of_processes[:split_places[i]])
                list_of_processes_B += [x for x in self.list_of_processes[split_places[i]:] if x not in list_of_processes_B]

            list_of_processes_A += [x for x in other.list_of_processes if x not in list_of_processes_A]
            list_of_processes_B += [x for x in self.list_of_processes if x not in list_of_processes_B]

            for process in list_of_processes_A:
                genotype_for_A[process] = sorted_setup[process]

            for process in list_of_processes_B:
                genotype_for_B[process] = sorted_setup[process]

            return FactoryQueue(genotype_for_A, 0, list_of_processes_A), FactoryQueue(genotype_for_B, 0,
                                                                                      list_of_processes_B)

        def calculate_time(self):
            working_setup = copy.deepcopy(self.genotype)
            current_time = 0
            current_machines_working = set()
            processes_running = dict()
            processes_done = set()
            processes_running_loop = dict()

            while len(processes_done) != len(working_setup):
                # if process is not running and its possible for it to run, then add its machine and process to proper sets
                for process in working_setup.keys():

                    list_of_machines = working_setup[process]
                    for machine in working_setup[process]:

                        machine_index = list_of_machines.index(machine)
                        if machine_index == 0:
                            previous_machine_count_number = 0
                        elif machine_index != 0:
                            previous_machine_count_number = list_of_machines[machine_index - 1][1]
                        if previous_machine_count_number != 0:
                            break

                        if (machine[0] not in current_machines_working) \
                                and (machine[1] != 0) \
                                and (process not in processes_running):
                            current_machines_working.add(machine[0])
                            processes_running[process] = machine[0]
                            processes_running_loop[process] = machine[0]
                            break
                        else:
                            pass

                # if process was not done in previous iteration add it to next loop
                for process in processes_running:
                    if process not in processes_running_loop:
                        processes_running_loop[process] = processes_running[process]
                    else:
                        pass

                # taking a step
                while len(processes_running_loop.keys()) != 0:

                    for process in working_setup.keys():
                        if process in processes_running_loop.keys():
                            for machine in working_setup[process]:
                                if machine[1] == 0:
                                    continue
                                else:
                                    if machine[0] in current_machines_working:
                                        machine[1] = machine[1] - 1
                                        processes_running_loop.pop(process)
                                        if machine[1] == 0:
                                            current_machines_working.remove(machine[0])
                                            processes_running.pop(process)

                                break

                # if process is done then add it to proper set
                for process in working_setup.keys():
                    if working_setup[process][-1][1] != 0:
                        break
                    else:
                        processes_done.add(process)

                current_time += 1
            self.fenotype = current_time

        def mutate(self):
            mutation_list = []
            weights = []
            mutated = []

            for i in range(num_mut_places + 1):
                mutation_list.append(i)
            for i in mutation_list:
                if i == 0:
                    weights.append(1 - mut_chance)
                else:
                    weights.append(mut_chance / (len(mutation_list) - 1))

            how_many_muts = rand.choices(mutation_list, weights=weights)
            how_many_muts = how_many_muts[0]

            while how_many_muts != 0:
                point_mut = rand.randint(0, len(self.list_of_processes) - 1)
                second_point_mut = rand.randint(0, len(self.list_of_processes) - 1)

                if point_mut not in mutated:
                    if point_mut != second_point_mut:
                        self.list_of_processes[point_mut], self.list_of_processes[second_point_mut] = \
                            self.list_of_processes[second_point_mut], self.list_of_processes[point_mut]
                        how_many_muts = how_many_muts - 1
                        mutated.extend([point_mut, second_point_mut])
                else:
                    continue

            new_genotype = {}
            for process in self.list_of_processes:
                new_genotype[process] = self.genotype[process]
            self.genotype = new_genotype.copy()

    def generate_population():
        adam = FactoryQueue(sorted_setup, 0, list(sorted_setup.keys()))
        adam.calculate_time()

        for _ in range(init_pop):
            population.append(copy.deepcopy(adam))
        return population

    def pairing(population):
        new_population = []

        for i in range(0, len(population), 2):
            if rand.random() <= cross_chance:
                a = population[i] + population[i + 1]
                new_population.append(a[0])
                new_population.append(a[1])
            else:
                new_population.append(population[i])
                new_population.append(population[i + 1])

        return new_population

    def natural_selection(population, ruletka=False):
        if ruletka:
            half = (len(population) * selection_surv_percent)
            fenotypesum = 0
            weights = []
            new_population = []

            for i in population:
                fenotypesum += i.fenotype
            for i in population:
                try:
                    weights.append((i.fenotype) / fenotypesum)
                except:
                    weights = None

            while len(new_population) != half:
                winner = np.random.choice(population, replace=False, p=weights)
                new_population.append(winner)

            while len(new_population) != len(population):
                new_population.append(rand.choices(new_population))

            np.random.shuffle(new_population)

        else:
            best = []
            half = (len(population) * selection_surv_percent)
            new_population = sorted(population, key=lambda x: x.fenotype, reverse=False)
            newer_bacteria = new_population[:int(half)]
            while len(best) != len(population):
                best.append(np.random.choice(newer_bacteria, replace=True))
            np.random.shuffle(best)

        return best

    population = generate_population()

    for i in range(epochs):

        for j in population:
            if rand.random() <= mut_chance:
                j.mutate()
            else:
                pass

        population = pairing(population)

        for j in population:
            j.calculate_time()

        population = natural_selection(population, False)

        suma = 0
        std = []
        for j in population:
            suma += j.fenotype
            std.append(j.fenotype)
        print(f'Run {i + 1}')
        # print(f'Best specimen genotype: {min(population, key=lambda x: x.fenotype).list_of_processes}')
        # print(f'Best specimen score: {min(population, key=lambda x: x.fenotype).fenotype}')
        # print(f'Mean score: {suma / len(population)}, with standard deviation: {np.std(std)}')
        # print('-------------------------------------------')

    return population


init_pop = [10, 20, 30]
population = []

mut_chance = [0.1, 0.2, 0.3]
num_mut_places = 20

cross_chance = 0.9
num_cross_places = [1, 5, 10]

selection_surv_percent = 0.5

epochs = 50

scores = pd.DataFrame(columns=['Population number', 'Mutation chance', 'Number of cross places',
                               'Mean score', 'Best specimen score', 'Best specimen genotype'])

for ip in init_pop:
    for mc in mut_chance:
        for ncp in num_cross_places:
            print(ip, mc, ncp)

            population = []
            evolved_population = complete_simulation(ip, population, mc, num_mut_places,
                                                     cross_chance, ncp, selection_surv_percent, epochs)
            suma = 0
            std = []
            for j in evolved_population:
                suma += j.fenotype
                std.append(j.fenotype)

            best_in_run = min(evolved_population, key=lambda x: x.fenotype)
            scores.loc[len(scores.index)] = [ip, mc, ncp, f'{suma / len(evolved_population)} ± {np.std(std)}',
                                             best_in_run.fenotype, best_in_run.list_of_processes]

scores.to_csv("scores_" + str(int(time.time())) + ".csv")
scores.to_excel("scores_" + str(int(time.time())) + ".xlsx")
