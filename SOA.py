import numpy as np
import math
from copy import deepcopy

population_size = 50
dimension = 10
range0 = -10
range1 = 10
epochs = 100

def initialize_population():
    return np.random.uniform(range0, range1, (population_size, dimension))


def initialize_params():

    f_c = 2
    u = 1
    v = 1

    return f_c, u, v


def initialize_best():
    best_fitness = 1000000
    best_agent = np.random.uniform(range0, range1, dimension)

    return best_fitness, best_agent


def get_gitness(particle):
    return np.sum(particle**2)


def update_best_agent_and_fitness(population, best_fitness, best_agent):
    for i in range(population_size):
        fitness_i = get_gitness(population[i])
        # if fitness_i <= best_fitness:
        best_fitness = fitness_i
        best_agent = population[i]

    return best_fitness, best_agent

def evaluate_population(population):
    population = np.maximum(population, range0)
    population = np.minimum(population, range1)
    return population


def run_SOA():

    population = initialize_population()
    # print(population.shape)
    f_c, u, v = initialize_params()
    best_fitness, best_agent = initialize_best()
    rd = np.random.uniform(0, 1)
    k = np.random.uniform(0, 2*np.pi)

    for epoch in range(epochs):

        best_fitness, best_agent = deepcopy(update_best_agent_and_fitness(population, best_fitness, best_agent))
        print("epoch: {} and best_fitness: {}".format(epoch, best_fitness))

        A = f_c - (epoch*f_c/epochs)
        # print(A)
        B = 2*A*A*rd

        ### migration ###
        C_s = A*population
        M_s = B*(best_agent - population)
        D_s = np.abs(C_s + M_s)
        print(D_s)

        ### attacking ###
        r = u*math.pow(math.e, k*v)
        x = r*math.cos(k)
        y = r*math.sin(k)
        z = r*k

        new_population = D_s*x*y*z + best_agent
        population = deepcopy(evaluate_population(new_population))
        print(best_agent)


def main():
    run_SOA()


if __name__ == "__main__":
    main()