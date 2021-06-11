from .ProxylessSupernet import ProxylessSupernet
from ofa.model_zoo import ofa_net
from ofa.tutorial import FLOPsTable, LatencyTable
from .LatencyPredictors.ofa_lat_predictor import OFA_NORM_CONSTANTS, load_ofa_mbv3_op_graph_gpu_lat_predictor, \
    load_ofa_mbv3_op_graph_cpu_lat_predictor, load_ofa_mbv3_op_graph_npu_lat_predictor


class OFASupernet(ProxylessSupernet):
    def __init__(self, logger=print, metrics=None, imagenet_path='/data/ImageNet/', device='cpu', **kwargs):
        super().__init__(logger=logger, metrics=metrics, imagenet_path=imagenet_path, device=device, ofa=True, **kwargs)

        # Can't currently run in-house latency predictors
        # self.metrics = ["accuracy", "FLOPS", "GPU_latency", "CPU_latency", "note10_latency", "NPU_latency"]
        self.metrics = ["accuracy", "FLOPS", "note10_latency", "NPU_latency"]

        self.resolution = kwargs.get('resolution', 224)

        # Accuracy supernet
        model_string = "ofa_mbv3_d234_e346_k357_w1." + str(kwargs.get('width', 0))
        self.logger("Model string is: {}".format(model_string))
        self.model = ofa_net(model_string, pretrained=True)

        # FLOPS and Note10 Latency predictors
        self.FLOPS_table = FLOPsTable(device=device, batch_size=1)
        self.lat_table = LatencyTable(resolutions=(self.resolution,))

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
            "resolution": self.resolution
        }

        self.logger("Overall architecture constraints: {}".format(self.overall_constraints))

    def flops_measure(self, architectures):
        return [self.FLOPS_table.predict_efficiency(arch) for arch in architectures]

    def note10_latency_measure(self, architectures):
        return [self.lat_table.predict_efficiency(arch) for arch in architectures]
