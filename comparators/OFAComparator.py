import numpy as np
from copy import deepcopy


# TODO base class and inheritence
class OFAComparator:
    def __init__(self, search_space, n=10, tau=0.95, K=5):
        self.search_space = search_space
        self.logging = search_space.logger
        self.n = n
        self.tau = tau
        self.K = K

        # Set depth of network to be max
        #self.search_space.depths = [np.max(self.search_space.depths)]
        #self.search_space.resolutions = (np.max(self.search_space.resolutions),)

    def execute(self, block):
        archs = self.search_space.random_sample(n=self.n)
        operations = self._gen_operations(block)
        # Matrix should be # Ops x n
        columns = len(archs)
        rows = len(operations)

        self.logging("Sampling %d random architectures and fixing Stage %d Block %d" % (self.n, block[0], block[1]))
        if block[1] > 1:
            self.logging("NOTE: As Block %d is being fixed, this block, or the one preceding it, will be the final "
                         "block of the stage" % block[1])

        self.tau_matrix = np.zeros((rows, columns))

        for j, arch in enumerate(archs):
            # TODO better to evaluate entire arch list at once. One list, then reshape
            # Run through all possible operations
            accuracies = self._evaluate_all_configs(arch, operations)
            # Record in matrix
            self.tau_matrix[:, j] = accuracies

        self.logging("Block, acc mu, acc sig, acc min, acc max, ")
        block_mu_list = self.tau_matrix.mean(axis=1).tolist()
        block_sig_list = self.tau_matrix.std(axis=1).tolist()
        block_min_list = self.tau_matrix.min(axis=1).tolist()
        block_max_list = self.tau_matrix.max(axis=1).tolist()
        for i, op in enumerate(operations):
            msg = "%s, %2.5f, %2.5f, %2.5f, %2.5f, " % (self._block_name(op),
                                                        block_mu_list[i],
                                                        block_sig_list[i],
                                                        block_min_list[i],
                                                        block_max_list[i])
            self.logging(msg)

        quantile = np.quantile(self.tau_matrix, self.tau)
        self.tau_matrix[self.tau_matrix < quantile] = 0
        self.tau_matrix[self.tau_matrix >= quantile] = 1
        self.tau_list = self.tau_matrix.sum(axis=0).tolist()
        k_best_indices = self._top_K(deepcopy(self.tau_list))

        self.logging("Best Top-%d architectures regardless of block fix:" % self.K)
        for k in range(self.K):
            self.logging("%d: %s" % (k + 1, self._arch_trim(archs[k_best_indices[k]], block)))

    def _gen_operations(self, block):
        ops = []
        #for stage in range(self.search_space.num_stages):
        #    for block in range(self.search_space.depths[0]):
        # If we're dealing with one of the last 2 blocks
        if block[1] > 1:
            ops.append([block[0], block[1], -1, -1])
        for expansion in self.search_space.expand_ratios:
            for kernel in self.search_space.kernel_sizes:
                ops.append([block[0], block[1], expansion, kernel])
        return ops

    def _evaluate_all_configs(self, arch, ops):
        arch_configs = []
        for op in ops:
            arch_copy = deepcopy(arch)
            arch_copy = self._arch_edit(arch_copy, op)
            arch_configs.append(arch_copy)
        return self.search_space.accuracy_measure(arch_configs)

    @staticmethod
    def _arch_edit(arch, op):
        if op[2] == -1 or op[3] == -1:
            arch['d'][op[0]] = op[1] + 1
        else:
            index = (4 * op[0]) + op[1]
            arch['e'][index] = op[2]
            arch['ks'][index] = op[3]
        return arch

    @staticmethod
    def _arch_trim(arch, block):
        index = (4 * block[0]) + block[1]
        arch['e'][index] = -1
        arch['ks'][index] = -1
        if block[1] > 1:
            arch['d'][block[0]] = block[1] + 1
        return arch

    def _top_K(self, tau_list):
        best_indices = []
        for k in range(self.K):
            index = tau_list.index(max(tau_list))
            tau_list[index] = 0
            best_indices.append(index)
        return best_indices

    def _block_name(self, block):
        assert len(block) == 4
        msg = "S%dB%d " % (block[0], block[1])
        if block[2] == -1 or block[3] == -1:
            msg += "Empty"
        else:
            msg += "MBConv%d_%dx%d" % (block[2], block[3], block[3])
        return msg
