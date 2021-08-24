from .ProxylessSupernet import ProxylessSupernet
from ofa.model_zoo import ofa_net
import random
from .LatencyPredictors.ofa_lat_predictor import OFA_NORM_CONSTANTS, load_ofa_resnet_op_graph_lat_predictor, ofa_resnet_op_graph_lat_predict_batch


class ResNet50Supernet(ProxylessSupernet):
    def __init__(self, logger=print, metrics=None, imagenet_path='/data/ImageNet/', device='cpu', **kwargs):
        super().__init__(logger=logger, imagenet_path=imagenet_path, device=device, ofa=True, **kwargs)

        # Currently cannot run in-house predictors for ResNet50
        #self.metrics = ["accuracy", "FLOPS", "GPU_Latency", "CPU_Latency"]
        self.metrics = ["accuracy", "FLOPS"]
        # These parameters are different from the MBConv structures of OFA and ProxylessNAS
        # There are 4 stages in total.
        # Each stage has a maximum depth of 2

        # Depth - 5 elements
        # First element is always 0 or 2
        # If 0, there is "stem skipping", if 2, no "stem skipping"
        # Remaining elements pertain to stage depths, specifically 2 minus the depth element for that stage.
        # Its subtracted later. Higher values mean larger depth.
        self.depths = kwargs.get('depth', [0, 1, 2])

        # Expands - 18 elements
        # 4, 4, 6, 4 blocks per stage.
        self.expand_ratios = kwargs.get('expand_ratios', [0.2, 0.25, 0.35])
        self.blocks_per_stage = [4, 4, 6, 4]
        self.stage_start_idxs = [0, 4, 8, 14]

        # Widths - 6 elements
        # First and 2nd element of width relate to the stems.
        # First element determines first stem width: {0, 1} gives 24, 2 gives 32 channels.
        # Second element determines 2nd stem width 0->40, 1->48, 2->64 channels
        # Remaining elements correspond to each stage.
        self.width_mult_idx = kwargs.get('widths', [0, 1, 2])
        self.stem_channels = [[24, 24, 32],
                              [40, 48, 64]]

        # These are the actual number of channels per stage that correspond to
        # the width_mult_idx values. Those index the sublists of this list
        # each sublist corresponds to a stage
        self.widths = [[168, 208, 256],
                       [336, 408, 512],
                       [664, 816, 1024],
                       [1328, 1640, 2048]]

        # GPU Latency predictor and normalization constant
        # self.GPU_latency_predictor = load_ofa_resnet_op_graph_lat_predictor(
        #    'models/Latency/ofa_resnet_op_graph_gpu_lat_predictor_best.pt')
        # self.GPU_latency_constant = OFA_NORM_CONSTANTS["ofa_resnet_op_graph_gpu_lat"]

        # CPU Latency predictor and normalization constant
        # self.CPU_latency_predictor = load_ofa_resnet_op_graph_lat_predictor(
        #    'models/Latency/ofa_resnet_op_graph_cpu_lat_predictor_best.pt')
        # self.CPU_latency_constant = OFA_NORM_CONSTANTS["ofa_resnet_op_graph_cpu_lat"] * 1000

        # Select metrics
        if metrics is not None:
            self.metrics = [self.metrics[i] for i in metrics]
        self.logger("Selected metrics:{}".format(self.metrics))

        model_string = "ofa_resnet50"
        self.logger("Model string is: {}".format(model_string))
        self.model = ofa_net(model_string, pretrained=True)

        self.overall_constraints = {
            "depths": self.depths,
            "expand_ratios": self.expand_ratios,
            "blocks_per_stage": self.blocks_per_stage,
            "width_mult_idx": self.width_mult_idx
        }

        self.logger("Overall architecture constraints: {}".format(self.overall_constraints))

    def random_sample(self, n=10):
        archs = []
        for _ in range(n):
            d = [random.choice([min(self.depths), max(self.depths)])]
            d += random.choices(self.depths, k=4)
            w = random.choices(self.width_mult_idx, k=6)
            e = random.choices(self.expand_ratios, k=18)
            sample = {
                'd': d,
                'e': e,
                'w': w
            }
            archs.append(sample)
        return archs

    def block_sample(self, n=10, **kwargs):
        block_list = kwargs.get('block_list', [[1, 0, 7, 6, 224]])
        archs = self.random_sample(n=n)
        for block in block_list:
            for arch in archs:
                # Fix stem skipping
                # of the form [0, val]
                if block[0] == 0:
                    self._fix_stem_skip(arch, block)
                # Fix a stem ratio
                # Of the form [1, stem, val]
                elif block[0] == 1:
                    self._fix_stem_widths(arch, block)
                # Fix an expansion ratio
                # Of the form [num != {0, 1}, stage, block, ratio, width]
                else:
                    self._fix_expand_ratio(arch, block)
        return archs

    # Format for expansion and width ratio changes
    # Of the form [num, stage, block, ratio, width]
    # num - Not 0 or 1; signals to the code that we're fixing a block with an exp ratio
    #       and the corresponding stage of said block with a width
    # stage - specified stage
    # block - block within stage
    # ratio - The expansion ratio for the specific block
    # width - The width ratio for the stage
    def _fix_expand_ratio(self, arch, block):
        # If we don't have the depth, make it so.
        if arch['d'][block[1] + 1] + self.blocks_per_stage[block[1]] - 2 < block[2] + 1:
            arch['d'][block[1] + 1] = block[2] - self.blocks_per_stage[block[1]] + 3
        #if arch['d'][block[1] + 1] < block[2]:
        #    arch['d'][block[1] + 1] = block[2]
        base_index = self.stage_start_idxs[block[1]]
        arch['e'][base_index + block[2]] = block[3]
        arch['w'][block[1] + 2] = block[4]

    # Format for fixing the stem widths
    # Of the form [1, stem, val]
    # 1 - Signals to the code that we're fixing the stems
    # stem - stem to fix, 0 or 1
    # val - value to take, 0, 1, or 2
    def _fix_stem_widths(self, arch, block):
        arch['w'][block[1]] = block[2]

    # Format for fixing stem skips
    # Of the form [0, val]
    # Where 0 signals to the code that this is the desired fix
    # Val determines whether to enable or disable
    # If val is equal or less than min(self.depths), stem skipping is enabled
    # Otherwise, it is disabled.
    def _fix_stem_skip(self, arch, block):
        if block[1] <= min(self.depths):
            arch['d'][0] = min(self.depths)
        else:
            arch['d'][0] = max(self.depths)

    def all_blocks(self):
        # Fix stem skip first
        blocks = [[[0, min(self.depths)]]]
        if max(self.depths) > min(self.depths):
            blocks.append([[0, max(self.depths)]])

        # Stem widths
        # First stem channel width 24
        blocks.append([[1, 0, 0]])
        # First stem channel width 32
        blocks.append([[1, 0, 2]])
        # Second stem channel widths 40, 48, 64
        blocks.append([[1, 1, 0]])
        blocks.append([[1, 1, 1]])
        blocks.append([[1, 1, 2]])

        # Stage widths and block expansion ratios
        for stage in range(4): # Stages
            for w in self.width_mult_idx:
                for b in range(self.blocks_per_stage[stage]):
                    for e in self.expand_ratios:
                        # Of the form [num, stage, block, ratio, width]
                        block = [[2, stage, b, e, w]]
                        blocks.append(block)
        self.logger("Number of block configurations: %d" % len(blocks))
        return blocks

    def block_meaning(self, **kwargs):
        block_list = kwargs.get('block_list', [[0, 0]])
        sorted(block_list, key=lambda x: x[0])

        def _translate_stem_skip(spec):
            if spec[1] <= min(self.depths):
                return "Stem skipping enabled"
            return "Step skipping disabled"

        def _translate_stem_width(spec):
            return "Stem %d with channels %d" % (spec[1] + 1,
                                                 self.stem_channels[spec[1]][spec[2]])

        def _translate_expansion(spec):
            return "U%d with %d C and L%d with ratio %f" % (spec[1],
                                                            self.widths[spec[1]][spec[4]],
                                                            spec[2],
                                                            spec[3])
        specification = ""
        for block in block_list:
            if block[0] == 0:
                specification += _translate_stem_skip(block)
            elif block[0] == 1:
                specification += _translate_stem_width(block)
            else:
                specification += _translate_expansion(block)
            specification += ", "
        return specification

    def set_arch_in_net(self, arch):
        self.model.set_active_subnet(d=arch['d'], e=arch['e'], w=arch['w'])

    def _lat_measure(self, archs, model, constant):
        net_configs = []
        resolutions = []
        for arch in archs:
            net_configs.append((arch['d'], arch['e'], arch['w']))
            resolutions.append(224)
        return ofa_resnet_op_graph_lat_predict_batch(net_configs, resolutions, model, constant,
                                                     batch_size=2, supernet=self.model)

    def gpu_latency_measure(self, architectures):
        return self._lat_measure(architectures, self.GPU_latency_predictor, self.GPU_latency_constant)

    def cpu_latency_measure(self, architectures):
        return self._lat_measure(architectures, self.CPU_latency_predictor, self.CPU_latency_constant)
