from .BaseSpace import *
import random
from ofa.tutorial import AccuracyPredictor, FLOPsTable, LatencyTable
from .LatencyPredictors.ofa_lat_predictor import OFA_NORM_CONSTANTS, load_ofa_mbv3_op_graph_gpu_lat_predictor, \
    load_ofa_mbv3_op_graph_cpu_lat_predictor, load_ofa_mbv3_op_graph_npu_lat_predictor, ofa_op_graph_lat_predict_batch


class OFAPredictorSpace(BaseSpace):
    def __init__(self, logger=print, metrics=None, device='cpu', **kwargs):
        super().__init__(logger)

        # Can't currently run in-house latency predictors
        # self.metrics = ["accuracy", "FLOPS", "GPU_latency", "CPU_latency", "note10_latency", "NPU_latency"]
        self.metrics = ["accuracy", "FLOPS", "note10_latency", "NPU_latency"]
        self.acc_predictor = AccuracyPredictor(
            pretrained=True,
            device=device
        )

        self.num_blocks = kwargs.get('num_blocks', 20)
        self.num_stages = kwargs.get('stages', 5)
        self.kernel_sizes = kwargs.get('kernel_sizes', [3, 5, 7])
        self.expand_ratios = kwargs.get('expand_ratios', [3, 4, 6])
        self.depths = kwargs.get('depths', [2, 3, 4])
        self.resolutions = kwargs.get('resolutions', (224,))  # Using kwargs you can change this to include
        # resolutions (160, 176, 192, 208, 224,) which covers the full OFA spectrum.

        # FLOPS and Note10 Latency predictors
        self.FLOPS_table = FLOPsTable(device=device, batch_size=1)
        self.lat_table = LatencyTable(resolutions=self.resolutions)

        # GPU Latency predictor and normalization constant
        # self.GPU_latency_predictor = load_ofa_mbv3_op_graph_gpu_lat_predictor()
        # self.GPU_latency_constant = OFA_NORM_CONSTANTS["ofa_mbv3_op_graph_gpu_lat"]

        # CPU Latency predictor and normalization constant
        # self.CPU_latency_predictor = load_ofa_mbv3_op_graph_cpu_lat_predictor()
        # self.CPU_latency_constant = OFA_NORM_CONSTANTS["ofa_mbv3_op_graph_cpu_lat"] * 1000

        # NPU Latency predictor and normalization constant
        self.NPU_latency_predictor = load_ofa_mbv3_op_graph_npu_lat_predictor()
        self.NPU_latency_constant = OFA_NORM_CONSTANTS["ofa_mbv3_op_graph_npu_lat"] / 1000

        # Subspace name and mbconv version - used for shared/inherited functions
        self.sub_space = "mbv3"
        self.mbv = 3

        # Select metrics
        if metrics is not None:
            self.metrics = [self.metrics[i] for i in metrics]
        self.logger("Selected metrics:{}".format(self.metrics))

        self.overall_constraints = {
            "num_blocks": self.num_blocks,
            "num_stages": self.num_stages,
            "kernel_sizes": self.kernel_sizes,
            "expand_ratios": self.expand_ratios,
            "depths": self.depths,
            "resolutions": self.resolutions
        }

        self.logger("Overall architecture constraints: {}".format(self.overall_constraints))

    # This file is taken from the original OFA repo, just modified to loop and give a number of archs
    def random_sample(self, n=10):
        archs = []
        for _ in range(n):

            d = []
            e = []
            ks = []
            for i in range(self.num_stages):
                d.append(random.choice(self.depths))

            for i in range(self.num_blocks):
                e.append(random.choice(self.expand_ratios))
                ks.append(random.choice(self.kernel_sizes))

            sample = {
                'wid': None,
                'ks': ks,
                'e': e,
                'd': d,
                'r': [random.choice(self.resolutions)]
            }
            archs.append(sample)
        return archs

    # kwargs should contain one parameter for fixing certain blocks:
    # block_list: List of lists. Length of outer list determines how many blocks to fix (usually 1)
    # Each inner list contains five values (stage, block, kernel, expand, res), referring to
    # - 0, stage: The stage (0-4) where the fixed block is located.
    # - 1, block: The block (0-3) within the stage.
    # - 2, kernel: Kernel size (3, 5, 7) of fixed block
    # - 3, expand: Expansion ratio (3, 4, 6) of fixed block
    # - 4, res: Input resolution for the fixed network. (Optional, if not given a random resolution will be chosen)
    # The resolution of later specifications takes precedence over earlier ones.
    def block_sample(self, n=10, **kwargs):
        # Default to making the first block of the second stage an MBConv6_7x7 with resolution 224
        block_list = kwargs.get('block_list', [[1, 0, 7, 6, 224]])
        archs = self.random_sample(n=n)
        for block in block_list:
            for arch in archs:
                # If the stage isn't deep enough, make it so.
                if arch['d'][block[0]] < block[1]:
                    arch['d'][block[0]] = block[1]
                # Now set the kernel size and expansion ratio
                base_index = 4 * block[0]
                arch['ks'][base_index + block[1]] = block[2]
                arch['e'][base_index + block[1]] = block[3]
                # Now set the resolution - if applicable:
                if len(block) > 4:
                    arch['r'] = [block[4]]
        return archs

    @staticmethod
    def _add_res(resolution):
        return " with Resolution %d" % resolution

    def block_meaning(self, **kwargs):
        block_list = kwargs.get('block_list', [[1, 0, 7, 6, 224]])
        specification = "Unit %d Layer %d as MBConv%d_%dx%d" % (block_list[0][0],
                                                                 block_list[0][1],
                                                                 block_list[0][3],
                                                                 block_list[0][2],
                                                                 block_list[0][2])

        if len(block_list[0]) > 4:
            specification += self._add_res(block_list[0][4])

        if len(block_list) > 1:
            for block in block_list[1:]:
                specification += ", "
                specification += "Unit %d Layer %d as MBConv%d_%dx%d" % (block[0],
                                                                         block[1],
                                                                         block[3],
                                                                         block[2],
                                                                         block[2])
                if len(block) > 4:
                    specification += self._add_res(block[4])

        return specification + ", "

    def accuracy_measure(self, architectures):
        results = self.acc_predictor.predict_accuracy(architectures).tolist()
        return [item * 100 for sublist in results for item in sublist]

    def flops_measure(self, architectures):
        return [self.FLOPS_table.predict_efficiency(arch) for arch in architectures]

    def note10_latency_measure(self, architectures):
        return [self.lat_table.predict_efficiency(arch) for arch in architectures]

    def _lat_measure(self, archs, model, constant):
        net_configs = []
        resolutions = []
        for arch in archs:
            net_config, res = self._convert_ofa_dict_to_list(arch)
            net_configs.append(net_config)
            resolutions.append(res)
        return ofa_op_graph_lat_predict_batch(net_configs, resolutions, model,
                                              norm_constant=constant, sub_space=self.sub_space)

    def gpu_latency_measure(self, architectures):
        return self._lat_measure(architectures, self.GPU_latency_predictor, self.GPU_latency_constant)

    def cpu_latency_measure(self, architectures):
        return self._lat_measure(architectures, self.CPU_latency_predictor, self.CPU_latency_constant)

    def npu_latency_measure(self, architectures):
        return self._lat_measure(architectures, self.NPU_latency_predictor, self.NPU_latency_constant)

    def _convert_ofa_dict_to_list(self, architecture):
        arch_list = []
        for stage_idx, depth in enumerate(architecture['d']):
            blk_idx = stage_idx * 4

            stage_list = []
            for blk in range(depth):
                expansion = architecture['e'][blk_idx + blk]
                kernel_sz = architecture['ks'][blk_idx + blk]

                block_op = "mbconv%d_e%d_k%d" % (self.mbv, expansion, kernel_sz)
                stage_list.append(block_op)
            arch_list.append(stage_list)
        return arch_list, architecture['r'][0]

    def all_blocks(self):
        blocks = []
        for res in self.resolutions:
            for stage in range(self.num_stages):
                for b in range(max(self.depths)):
                    for e in self.expand_ratios:
                        for k in self.kernel_sizes:
                            if len(self.resolutions) == 1:
                                block = [[stage, b, k, e]]
                            else:
                                block = [[stage, b, k, e, res]]
                            blocks.append(block)
        self.logger("Number of block configuarions: %d" % len(blocks))
        return blocks
