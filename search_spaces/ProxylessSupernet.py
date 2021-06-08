from .BaseSpace import *
from misc.imagenet_loader import get_imagenet_val_loader, get_imagenet_calib_loader, get_imagenet_RAM_val_loader
from ofa.model_zoo import ofa_net
from ofa.utils import AverageMeter, accuracy
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
import torch.backends.cudnn as cudnn
import torch
import random
from tqdm import tqdm
from torchprofile import profile_macs
from .LatencyPredictors.ofa_lat_predictor import OFA_NORM_CONSTANTS, ofa_op_graph_lat_predict_batch


class ProxylessSupernet(BaseSpace):
    def __init__(self, logger=print, metrics=None, imagenet_path='/data/ImageNet/', device='cpu', fast=False,
                 ofa=False, **kwargs):
        super().__init__(logger)

        self.inet_path = imagenet_path
        self.batch_size = kwargs.get('batch_size', 250)
        self.device = device

        self.ofa = ofa
        self.num_blocks = kwargs.get('num_blocks', 21)
        self.num_stages = kwargs.get('stages', 6)
        if self.ofa:
            self.num_blocks = 20
            self.num_stages = 5
        self.kernel_sizes = kwargs.get('kernel_sizes', [3, 5, 7])
        self.expand_ratios = kwargs.get('expand_ratios', [3, 4, 6])
        self.depths = kwargs.get('depths', [2, 3, 4])
        self.resolution = kwargs.get('resolution', 224)

        if metrics is None or 0 in metrics:
            workers = kwargs.get('workers', 8)
            if fast:
                self.logger("Using faster RAM validation data loader")
                self.validation_loader = get_imagenet_RAM_val_loader(batch_size=self.batch_size,
                                                                     num_workers=workers,
                                                                     res=self.resolution,
                                                                     imagenet_path="models/ImageNetRAM/")
            else:
                self.logger("Using traditional disk validation data loader")
                self.validation_loader = get_imagenet_val_loader(imagenet_path=imagenet_path,
                                                                 batch_size=self.batch_size,
                                                                 workers=workers,
                                                                 size=self.resolution)

            self.calibration_loader = get_imagenet_calib_loader(imagenet_path=imagenet_path,
                                                                batch_size=self.batch_size,
                                                                workers=workers,
                                                                size=self.resolution)

        if not ofa:
            from .LatencyPredictors.ofa_lat_predictor import load_ofa_pn_op_graph_gpu_lat_predictor, \
                load_ofa_pn_op_graph_cpu_lat_predictor, load_ofa_pn_op_graph_npu_lat_predictor

            # Currently cannot run in-house predictors
            # self.metrics = ["accuracy", "FLOPS", "GPU_latency", "CPU_latency", "NPU_latency"]
            self.metrics = ["accuracy", "FLOPS"]

            model_string = "ofa_proxyless_d234_e346_k357_w1.3"
            self.logger("Model string is: {}".format(model_string))
            self.model = ofa_net(model_string, pretrained=True)

            # GPU Latency predictor and normalization constant
            self.GPU_latency_predictor = load_ofa_pn_op_graph_gpu_lat_predictor()
            self.GPU_latency_constant = OFA_NORM_CONSTANTS["ofa_pn_op_graph_gpu_lat"]

            # CPU Latency predictor and normalization constant
            self.CPU_latency_predictor = load_ofa_pn_op_graph_cpu_lat_predictor()
            self.CPU_latency_constant = OFA_NORM_CONSTANTS["ofa_pn_op_graph_cpu_lat"] * 1000

            # NPU Latency predictor and normalization constant
            self.NPU_latency_predictor = load_ofa_pn_op_graph_npu_lat_predictor()
            self.NPU_latency_constant = OFA_NORM_CONSTANTS["ofa_pn_op_graph_npu_lat"] / 1000

            # Subspace name and mbconv version - used for shared/inherited functions
            self.sub_space = "pn"
            self.mbv = 2

            self.overall_constraints = {
                "num_blocks": self.num_blocks,
                "num_stages": self.num_stages,
                "kernel_sizes": self.kernel_sizes,
                "expand_ratios": self.expand_ratios,
                "depths": self.depths,
                "resolution": self.resolution
            }

            self.logger("Overall architecture constraints: {}".format(self.overall_constraints))

            # Select metrics
            if metrics is not None:
                self.metrics = [self.metrics[i] for i in metrics]
            self.logger("Selected metrics:{}".format(self.metrics))

    # This file is taken from the original OFA repo, just modified to loop and give a number of archs
    def random_sample(self, n=10):
        archs = []
        for _ in range(n):

            d = []
            e = []
            ks = []
            for i in range(self.num_stages):
                d.append(random.choice(self.depths))
            if not self.ofa:
                d[-1] = 1

            for i in range(self.num_blocks):
                e.append(random.choice(self.expand_ratios))
                ks.append(random.choice(self.kernel_sizes))

            sample = {
                'ks': ks,
                'e': e,
                'd': d,
                'r': [self.resolution]
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

    # This function exists because the fields for ProxylessNAS and ResNet are different
    def set_arch_in_net(self, arch):
        self.model.set_active_subnet(d=arch['d'], e=arch['e'], ks=arch['ks'])

    def accuracy_measure(self, architectures):
        top1s = []
        for arch in architectures:
            self.set_arch_in_net(arch)
            subnet = self.model.get_active_subnet().to(self.device)
            set_running_statistics(subnet, self.calibration_loader)
            top1 = self._validate(subnet)

            top1s.append(top1)
        return top1s

    def flops_measure(self, architectures):
        input_tensor = torch.randn(1, 3, self.resolution, self.resolution).to(self.device)
        flops_list = []
        for arch in architectures:
            self.set_arch_in_net(arch)
            subnet = self.model.get_active_subnet().to(self.device)
            macs = profile_macs(subnet, input_tensor)
            flops_list.append(macs / 500000)
        return flops_list

    def _lat_measure(self, archs, model, constant):
        net_configs = []
        resolutions = []
        for arch in archs:
            net_config, res = self._convert_ofa_dict_to_list(arch)
            net_configs.append(net_config)
            resolutions.append(res)
        return ofa_op_graph_lat_predict_batch(net_configs, resolutions, model, norm_constant=constant, sub_space=self.sub_space)

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
        for stage in range(self.num_stages):
            num_blocks = max(self.depths)
            if stage > 4:
                num_blocks = 1
            for b in range(num_blocks):
                for e in self.expand_ratios:
                    for k in self.kernel_sizes:
                        block = [[stage, b, k, e, self.resolution]]
                        blocks.append(block)
        self.logger("Number of block configurations: %d" % len(blocks))
        return blocks

    def _validate(self, net):
        cudnn.benchmark = True
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        net.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.no_grad():
            with tqdm(total=len(self.validation_loader), desc='Validate') as t:
                for i, (images, labels) in enumerate(self.validation_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    output = net(images)
                    loss = criterion(output, labels)
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
                    t.set_postfix({
                        'loss': losses.avg,
                        'top1': top1.avg,
                        'top5': top5.avg,
                        'img_size': images.size(2),
                    })
                    t.update(1)

        return top1.avg
