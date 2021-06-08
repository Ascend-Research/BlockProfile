import copy
import random
from tqdm import tqdm
import numpy as np
# TODO solve what happens when repeat architectures are in the final population

__all__ = ['EvolutionFinder']


class EvolutionFinder:

    def __init__(self, search_space, manager, constraint_type="FLOPS", constraint=500, logger=print, **kwargs):

        # Establish constraint type and default quantity
        self.constraint_type = str.lower(constraint_type)
        self.constraint = constraint
        self.manager = manager

        # Confirm that the search space can query the requested constraint
        assert hasattr(search_space, '%s_measure' % self.constraint_type), "Method does not have constraint measurement"
        self.constraint_func = getattr(search_space, '%s_measure' % self.constraint_type).__name__

        # Set search space and logging function
        self.search_space = search_space
        self.logger = logger

        # Print constrained search space.
        const_space = self.manager.describe_constraint()
        self.logger(const_space)

        # TODO this may need to change to accomodate ResNet, or it will need its own class.
        self.num_blocks = search_space.num_blocks
        self.num_stages = search_space.num_stages

        # Search-specific parameters
        self.mutate_prob = kwargs.get('mutate_prob', 0.1)
        self.population_size = kwargs.get('population_size', 100)
        self.max_time_budget = kwargs.get('max_time_budget', 500)
        self.parent_ratio = kwargs.get('parent_ratio', 0.25)
        self.mutation_ratio = kwargs.get('mutation_ratio', 0.5)

        self.mutation_numbers = int(round(self.mutation_ratio * self.population_size))
        self.parents_size = int(round(self.parent_ratio * self.population_size))

    def random_sample(self):
        while True:
            arch_sample = self.manager.random_sample()
            arch_efficiency = eval("self.search_space.%s([arch_sample])" % self.constraint_func)[0]
            if arch_efficiency <= self.constraint:
                return arch_sample, arch_efficiency

    def mutate_sample(self, arch):
        while True:
            new_arch = self.manager.mutate(arch, self.mutate_prob)
            arch_efficiency = eval("self.search_space.%s([new_arch])" % self.constraint_func)[0]
            if arch_efficiency <= self.constraint:
                return new_arch, self.constraint

    def crossover_sample(self, sample1, sample2):
        while True:
            new_arch = copy.deepcopy(sample1)
            for key in new_arch.keys():
                if not isinstance(new_arch[key], list):
                    continue
                for i in range(len(new_arch[key])):
                    new_arch[key][i] = random.choice([sample1[key][i], sample2[key][i]])

            arch_efficiency = eval("self.search_space.%s([new_arch])" % self.constraint_func)[0]
            if arch_efficiency <= self.constraint:
                return new_arch, arch_efficiency

    def run_evolution_search(self, constraint=500, verbose=False):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        self.constraint = constraint

        best_valids = [-100]
        population = []  # (validation, sample, latency) tuples  # TODO re-work the ordering its bad
        child_pool = []
        efficiency_pool = []
        best_info = None
        if verbose:
            self.logger('Generate random population...')
        for _ in range(self.population_size):
            sample, efficiency = self.random_sample()
            child_pool.append(sample)
            efficiency_pool.append(efficiency)

        accs = self.search_space.accuracy_measure(child_pool)
        for i in range(self.population_size):
            population.append((accs[i], child_pool[i], efficiency_pool[i]))

        if verbose:
            self.logger('Start Evolution...')
        # After the population is seeded, proceed with evolving the population.
        for generation in tqdm(range(self.max_time_budget),
                         desc='Searching with %s constraint (%s)' % (self.constraint_type, self.constraint)):
            parents = sorted(population, key=lambda x: x[0])[::-1][:self.parents_size]
            acc = parents[0][0]
            if verbose:
                self.logger('Iter: {} Acc: {}'.format(generation - 1, parents[0][0]))

            if acc > best_valids[-1]:
                best_valids.append(acc)
                best_info = parents[0]
            else:
                best_valids.append(best_valids[-1])

            population = parents
            child_pool = []
            efficiency_pool = []

            for i in range(self.mutation_numbers):
                par_sample = population[np.random.randint(self.parents_size)][1]
                # Mutate
                new_sample, efficiency = self.mutate_sample(par_sample)
                child_pool.append(new_sample)
                efficiency_pool.append(efficiency)

            for i in range(self.population_size - self.mutation_numbers):
                par_sample1 = population[np.random.randint(self.parents_size)][1]
                par_sample2 = population[np.random.randint(self.parents_size)][1]
                # Crossover
                new_sample, efficiency = self.crossover_sample(par_sample1, par_sample2)
                child_pool.append(new_sample)
                efficiency_pool.append(efficiency)

            accs = self.search_space.accuracy_measure(child_pool)
            for i in range(self.population_size):
                population.append((accs[i], child_pool[i], efficiency_pool[i]))

        self.logger("End of search population:")
        for arch in population:
            self.logger(arch)

        return best_valids, best_info
