from ofa.tutorial import AccuracyPredictor, FLOPsTable, LatencyTable
from .ProxylessPredictorSpace import ProxylessPredictorSpace
from .LatencyPredictors.ofa_lat_predictor import OFA_NORM_CONSTANTS, load_ofa_mbv3_op_graph_gpu_lat_predictor, \
    load_ofa_mbv3_op_graph_cpu_lat_predictor


class OFAPredictorSpace(ProxylessPredictorSpace):
    def __init__(self, logger=print, metrics=None, device='cpu', **kwargs):
        super().__init__(logger=logger, device=device, ofa=True, **kwargs)

        self.metrics = ["accuracy", "FLOPS", "GPU_latency", "CPU_latency", "note10_latency"]
        self.acc_predictor = AccuracyPredictor(
            pretrained=True,
            device=device
        )

        # FLOPS and Note10 Latency predictors
        self.FLOPS_table = FLOPsTable(device=device, batch_size=1)
        self.lat_table = LatencyTable(resolutions=self.resolutions)

        # GPU Latency predictor and normalization constant
        self.GPU_latency_predictor = load_ofa_mbv3_op_graph_gpu_lat_predictor()
        self.GPU_latency_constant = OFA_NORM_CONSTANTS["ofa_mbv3_op_graph_gpu_lat"]

        # CPU Latency predictor and normalization constant
        self.CPU_latency_predictor = load_ofa_mbv3_op_graph_cpu_lat_predictor()
        self.CPU_latency_constant = OFA_NORM_CONSTANTS["ofa_mbv3_op_graph_cpu_lat"] * 1000

        # Subspace name and mbconv version - used for shared/inherited functions
        self.sub_space = "mbv3"
        self.mbv = 3

        # Select metrics
        if metrics is not None:
            self.metrics = [self.metrics[i] for i in metrics]
        self.logger("Selected metrics:{}".format(self.metrics))

    # Overriding the implementation of flops_measure in ProxylessPredictorSpace as the
    # return of the FLOPS tables are different
    def flops_measure(self, architectures):
        return [self.FLOPS_table.predict_efficiency(arch) for arch in architectures]

    def note10_latency_measure(self, architectures):
        return [self.lat_table.predict_efficiency(arch) for arch in architectures]

    def block_meaning(self, **kwargs):
        return super().block_meaning(**kwargs).replace("Stage", "Unit").replace("Block", "Layer")
