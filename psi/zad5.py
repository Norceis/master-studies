import numpy as np
import random as rand

init_pop = 100
gene_num = 1
bacteria = []

mut_chance = 0.1
num_mut_places = 1

cross_chance = 0.8
num_cross_places = 1

selection_surv_percent = 0.05

epochs = 1000

def dectobin(n):
    return bin(n).replace("0b", "")

class Bacterium:

    def __init__(self, genotype, fenotype):
        self.genotype = genotype
        self.fenotype = fenotype

    def __str__(self):
        return "Bacterium G(" + str(self.genotype) + "), F(" + str(self.fenotype) + ")"

    def __repr__(self):
        return "Bacterium G(" + str(self.genotype) + "), F(" + str(self.fenotype) + ")"

    def __add__(self, other):
        splittedself = []
        splittedother = []
        splitplaces = []
        genotypeA = []
        genotypeB = []
        genotypeforA = []
        genotypeforB = []

        for i in range(len(self.genotype)):
            for j in self.genotype[i]:
                splittedself.append(j)
            for j in other.genotype[i]:
                splittedother.append(j)
        for i in range(num_cross_places):
            splitplaces.append(rand.randint(1,len(splittedself)-2))
        splitplaces.sort()

        for i in range(num_cross_places):
            genotypeA = splittedself[:splitplaces[i]] + splittedother[splitplaces[i]:]
            genotypeB = splittedother[:splitplaces[i]] + splittedself[splitplaces[i]:]

        for i in range(gene_num):
            genotypeforA.append(''.join(genotypeA[(i * 8):((i + 1) * 8)]))
            genotypeforB.append(''.join(genotypeB[(i * 8):((i + 1) * 8)]))

        # first.fenotype = 0
        # second.fenotype = 0
        # for i in range(len(first.genotype)):
        #     first.fenotype += int(first.genotype[i], 2)
        #     second.fenotype += int(second.genotype[i], 2)

        return Bacterium(genotypeforA, 0), Bacterium(genotypeforB, 0)

    def express(self):
        self.fenotype = 0
        for i in range(len(self.genotype)):
            self.fenotype += int(self.genotype[i], 2)
        return self


    def mutate(self):
        mutationlist = []
        weights = []
        splitted = []
        mutated = []
        for i in range(num_mut_places + 1):
            mutationlist.append(i)
        for i in mutationlist:
            if i == 0:
                weights.append(1 - mut_chance)
            else:
                weights.append(mut_chance / (len(mutationlist) - 1))

        howmanymuts = rand.choices(mutationlist, weights=weights)

        howmanymuts = howmanymuts[0]

        for i in range(len(self.genotype)):
            for j in self.genotype[i]:
                splitted.append(j)

        while howmanymuts != 0:
            pointmut = rand.randint(0, len(splitted)-1)

            if pointmut not in mutated:
                if splitted[pointmut] == '0':
                    splitted[pointmut] = '1'
                    howmanymuts = howmanymuts - 1
                else:
                    splitted[pointmut] = '0'
                    howmanymuts = howmanymuts - 1
                mutated.append(pointmut)
            else:
                continue

        for i in range(gene_num):
            self.genotype[i] = ''.join(splitted[(i*8):((i+1)*8)])


        self.fenotype = 0
        for i in range(len(self.genotype)):
            self.fenotype += int(self.genotype[i], 2)


def generatepopulation():
        for i in range(init_pop):
            genotype = []
            fenotype = 0
            for i in range(gene_num):
                genotype.append(dectobin(rand.randint(0,50)))
                while len(genotype[i]) != 8:
                    genotype[i] = '0' + genotype[i]
            for i in range(len(genotype)):
                fenotype += int(genotype[i],2)
            bacteria.append(Bacterium(genotype, fenotype))
        return bacteria

def pairing(bacteria):
    newpop = []
    for i in range(0, len(bacteria), 2):
        if rand.random() <= cross_chance:
            a = bacteria[i] + bacteria[i + 1]
            newpop.append(a[0])
            newpop.append(a[1])
        else:
            newpop.append(bacteria[i])
            newpop.append(bacteria[i+1])

    return newpop

def naturalselection(bacteria, ruletka=True):
    if ruletka:
        half = (len(bacteria)*selection_surv_percent)
        fenotypesum = 0
        weights = []
        newbacteria = []

        for i in bacteria:
            fenotypesum += i.fenotype
        for i in bacteria:
            try:
                weights.append((i.fenotype)/fenotypesum)
            except:
                weights = None

        while len(newbacteria) != half:
            winner = np.random.choice(bacteria, replace=False, p=weights)
            newbacteria.append(winner)

        while len(newbacteria) != len(bacteria):
            newbacteria.append(rand.choices(newbacteria))

        np.random.shuffle(newbacteria)

    else:
        best = []
        half = (len(bacteria)*selection_surv_percent)
        newbacteria = sorted(bacteria, key=lambda x: x.fenotype, reverse=True)

        newerbacteria = newbacteria[:int(half)]

        while len(best) != len(bacteria):
            best.append(np.random.choice(newerbacteria, replace=True))
        np.random.shuffle(best)
    return best


def checksum(bacterium):
    fenotype = 0
    splitted = []
    for i in range(len(bacterium.genotype)):
        splitted += [j for j in bacterium.genotype[i]]
    for i in range(len(bacterium.genotype)):
        fenotype += int(bacterium.genotype[i], 2)
    if fenotype == bacterium.fenotype:
        return True
    else:
        return False

bacteria = generatepopulation()

for i in range(epochs):

    # if i % 1 == 0:
    #     suma = 0
    #     std = []
    #     for j in bacteria:
    #         suma += j.fenotype
    #         std.append(j.fenotype)
    #
    #     print(i, '  ', suma/len(bacteria), ' ' , np.std(std), std)

    for j in bacteria:
        if rand.random() <= mut_chance:
            j.mutate()
        else:
            pass

    if i % 1 == 0:
        suma = 0
        std = []
        for j in bacteria:
            suma += j.fenotype
            std.append(j.fenotype)

        print(i, '  ', suma / len(bacteria), ' ', np.std(std), std)

    bacteria = pairing(bacteria)
    for j in bacteria:
        j.express()

    # if i % 1 == 0:
    #     suma = 0
    #     std = []
    #     for j in bacteria:
    #         suma += j.fenotype
    #         std.append(j.fenotype)
    # print(std)

    bacteria = naturalselection(bacteria, False)

    # if i % 1 == 0:
    #     suma = 0
    #     std = []
    #     for j in bacteria:
    #         suma += j.fenotype
    #         std.append(j.fenotype)
    # print(std)