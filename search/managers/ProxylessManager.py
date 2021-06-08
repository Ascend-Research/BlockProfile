import random
import copy


class ProxylessManager:
    def __init__(self, constraint):
        self.constraint = constraint

    def describe_constraint(self):
        message = "Constrained space:\n"
        for stage in range(len(self.constraint['d'])):
            message += "Stage {}, Depth: {}, Expansion: {}, Kernel: {}\n".format(stage,
                                                                               self.constraint['d'][stage],
                                                                               self.constraint['e'][stage],
                                                                               self.constraint['ks'][stage])
        message += "With Resolutions {}".format(self.constraint['r'])
        return message

    def random_sample(self):
        arch = {'d': [],
                'ks': [],
                'e': []}

        for stage in range(len(self.constraint['d'])):
            arch['d'].append(random.choice(self.constraint['d'][stage]))

            for block in range(4):
                arch['ks'].append(random.choice(self.constraint['ks'][stage]))
                arch['e'].append(random.choice(self.constraint['e'][stage]))
        arch['r'] = [random.choice(self.constraint['r'])]
        return arch

    def mutate(self, arch, prob=0.1):
        new_arch = copy.deepcopy(arch)

        # Alter resolution
        if len(self.constraint['r']) > 1 and random.random() < prob:
            new_arch['r'][0] = random.choice(self.constraint['r'])

        for stage in range(len(self.constraint['d'])):
            if random.random() < prob:
                new_arch['d'][stage] = random.choice(self.constraint['d'][stage])

            for block in range(4):
                if random.random() < prob:
                    new_arch['ks'][(stage * 4) + block] = random.choice(self.constraint['ks'][stage])
                if random.random() < prob:
                    new_arch['e'][(stage * 4) + block] = random.choice(self.constraint['e'][stage])
        return new_arch
