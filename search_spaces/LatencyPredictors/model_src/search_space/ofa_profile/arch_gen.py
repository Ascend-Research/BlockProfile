import torch
from .constants import *
from .arch_utils import get_final_channel_sizes
from search_spaces.LatencyPredictors.utils.model_utils import device
from search_spaces.LatencyPredictors.utils.math_utils import make_divisible


def get_ofa_pn_mbconv_io_shapes(net_configs, w=OFA_W_PN,
                                H=224, W=224, normalize=True,
                                H_max=224, W_max=224,
                                strides=(2, 2, 2, 1, 2, 1),
                                block_channel_sizes=(16, 24, 40, 80, 96, 192, 320),
                                log_f=print):
    assert len(net_configs) == len(strides)
    rv = []
    block_channel_sizes = get_final_channel_sizes(block_channel_sizes, w)
    H_in, W_in = H // 2, W // 2
    H_out, W_out = H_in, W_in
    C_in = block_channel_sizes[0]
    C_max = max(block_channel_sizes)
    H_max /= 2
    W_max /= 2
    for stage_i, blocks in enumerate(net_configs):
        C_out = block_channel_sizes[stage_i + 1]
        stride = strides[stage_i]
        stage_shapes = []
        for block_i, block in enumerate(blocks):
            if block_i == 0 and stride == 2:
                H_out = H_in // 2
                W_out = W_in // 2
            stage_shapes.append( [H_in, H_out, W_in, W_out, C_in, C_out] )
            C_in = C_out
            H_in = H_out
            W_in = W_out
        rv.append(stage_shapes)
    if normalize:
        if log_f is not None:
            log_f("Normalizing H_max={}".format(H_max))
            log_f("Normalizing W_max={}".format(W_max))
            log_f("Normalizing C_max={}".format(C_max))
        for stage_shapes in rv:
            for b_shapes in stage_shapes:
                b_shapes[0] /= H_max
                b_shapes[1] /= H_max
                b_shapes[2] /= W_max
                b_shapes[3] /= W_max
                b_shapes[4] /= C_max
                b_shapes[5] /= C_max
                assert all(0 <= v <= 1 for v in b_shapes), \
                    "Normalized block shapes: {}".format(b_shapes)
    return rv


def get_ofa_pn_net_idx_shape_feat(net_configs,
                                  H=224, W=224, normalize=True,
                                  H_max=224, W_max=224,
                                  log_f=print):
    shapes = get_ofa_pn_mbconv_io_shapes(net_configs,
                                         H=H, W=W, normalize=normalize,
                                         H_max=H_max, W_max=W_max,
                                         log_f=log_f)
    net_inds = []
    net_shapes = []
    for stage_i, blocks in enumerate(net_configs):
        for block_i, b_op in enumerate(blocks):
            type_idx = PN_OP2IDX[b_op]
            net_inds.append(type_idx)
            net_shapes.append(shapes[stage_i][block_i])
    assert len(net_inds) == len(net_shapes)
    return net_inds, net_shapes


def get_ofa_mbv3_mbconv_io_shapes(net_configs, w=OFA_W_MBV3,
                                  H=224, W=224, normalize=True,
                                  H_max=224, W_max=224,
                                  strides=(2, 2, 2, 1, 2),
                                  block_channel_sizes=(16, 24, 40, 80, 112, 160),
                                  log_f=print):
    assert len(net_configs) == len(strides)
    rv = []
    block_channel_sizes = get_final_channel_sizes(block_channel_sizes, w)
    H_in, W_in = H // 2, W // 2
    H_out, W_out = H_in, W_in
    C_in = block_channel_sizes[0]
    C_max = max(block_channel_sizes)
    H_max /= 2
    W_max /= 2
    for stage_i, blocks in enumerate(net_configs):
        C_out = block_channel_sizes[stage_i + 1]
        stride = strides[stage_i]
        stage_shapes = []
        for block_i, block in enumerate(blocks):
            if block_i == 0 and stride == 2:
                H_out = H_in // 2
                W_out = W_in // 2
            stage_shapes.append( [H_in, H_out, W_in, W_out, C_in, C_out] )
            C_in = C_out
            H_in = H_out
            W_in = W_out
        rv.append(stage_shapes)
    if normalize:
        if log_f is not None:
            log_f("Normalizing H_max={}".format(H_max))
            log_f("Normalizing W_max={}".format(W_max))
            log_f("Normalizing C_max={}".format(C_max))
        for stage_shapes in rv:
            for b_shapes in stage_shapes:
                b_shapes[0] /= H_max
                b_shapes[1] /= H_max
                b_shapes[2] /= W_max
                b_shapes[3] /= W_max
                b_shapes[4] /= C_max
                b_shapes[5] /= C_max
                assert all(0 <= v <= 1 for v in b_shapes), \
                    "Normalized block shapes: {}".format(b_shapes)
    return rv


def get_ofa_mbv3_net_idx_shape_feat(net_configs,
                                    H=224, W=224, normalize=True,
                                    H_max=224, W_max=224,
                                    log_f=print):
    shapes = get_ofa_mbv3_mbconv_io_shapes(net_configs,
                                           H=H, W=W, normalize=normalize,
                                           H_max=H_max, W_max=W_max,
                                           log_f=log_f)
    net_inds = []
    net_shapes = []
    for stage_i, blocks in enumerate(net_configs):
        for block_i, b_op in enumerate(blocks):
            type_idx = MBV3_OP2IDX[b_op]
            net_inds.append(type_idx)
            net_shapes.append(shapes[stage_i][block_i])
    assert len(net_inds) == len(net_shapes)
    return net_inds, net_shapes


def ofa_resnet_op2idx():
    # First add stems
    stem_output_channels = [
        make_divisible(64 * width_mult, 8) for width_mult in OFA_RES_WIDTH_MULTIPLIERS
    ]
    stem_hidden_channels = [
        make_divisible(channel // 2, 8) for channel in stem_output_channels
    ]
    blocks = []
    unique_blocks = set()
    for h in stem_hidden_channels:
        for o in stem_output_channels:
            for prefix in OFA_RES_STEM_PREFIXES:
                stem_block = "_".join([prefix, "h{}".format(h), "o{}".format(o)])
                if stem_block not in unique_blocks:
                    unique_blocks.add(stem_block)
                    blocks.append(stem_block)
    # Next add blocks
    for r in OFA_RES_EXPANSION_RATIOS:
        for k in OFA_RES_KERNEL_SIZES:
            if r < 1.0:
                exp_str = "e0{}".format(int(r * 100.))
            else:
                exp_str = "e{}".format(int(r * 100.))
            res_block = "_".join(["res", exp_str, "k{}".format(k)])
            blocks.append(res_block)
            assert res_block not in unique_blocks, "Duplicated name: {}".format(res_block)
            unique_blocks.add(res_block)
    return {b: i for i, b in enumerate(blocks)}


class InputOutputShapeHook:

    def __init__(self, module:torch.nn.Module):
        self.input_shape = None
        self.output_shape = None
        self.handle = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input_tsr, output_tsr):
        if isinstance(input_tsr, tuple):
            self.input_shape = tuple(input_tsr[0].shape)
        else:
            self.input_shape = tuple(input_tsr.shape)
        if isinstance(output_tsr, tuple):
            self.output_shape = tuple(output_tsr[0].shape)
        else:
            self.output_shape = tuple(output_tsr.shape)

    def close(self):
        self.handle.remove()


def _add_ofa_resnet_hooks(net):
    #from search_spaces.LatencyPredictors.model_src.search_space.ofa_profile.arch_gpu_cpu_lat import InputOutputShapeHook
    hooks = []
    hook = InputOutputShapeHook(net.input_stem[0])
    hooks.append(hook)
    hook = InputOutputShapeHook(net.input_stem[-1])
    hooks.append(hook)
    for bi, block in enumerate(net.blocks):
        hook = InputOutputShapeHook(block)
        hooks.append(hook)
    return hooks


def _get_ofa_resnet_subnet_io_shapes(subnet, resolution):
    hooks = _add_ofa_resnet_hooks(subnet)
    net_batch = torch.ones([1, 3, resolution, resolution]).float().to(device())
    with torch.no_grad():
        subnet.eval()
        subnet(net_batch)
    net_shapes = []
    # Special treatment for stem
    C_in, H_in, W_in = hooks[0].input_shape[1:]
    C_out, H_out, W_out = hooks[1].output_shape[1:]
    net_shapes.append((H_in, H_out, W_in, W_out, C_in, C_out))
    # For blocks
    for hook in hooks[2:]:
        C_in, H_in, W_in = hook.input_shape[1:]
        C_out, H_out, W_out = hook.output_shape[1:]
        net_shapes.append((H_in, H_out, W_in, W_out, C_in, C_out))
    for hook in hooks:
        hook.close()
    return net_shapes


def get_ofa_resnet_io_shapes(d_list, e_list, w_list, resolution,
                             loaded_model=None):
    from ofa.model_zoo import ofa_net
    from search_spaces.LatencyPredictors.model_src.search_space.ofa_profile.constants import OFA_RES_STAGE_MIN_N_BLOCKS
    if loaded_model is not None:
        #model_dir = os.sep.join([SAVED_MODELS_DIR, "ofa_checkpoints"])
        #if not os.path.isdir(model_dir):
        #    os.mkdir(model_dir)
        ofa_network = ofa_net('ofa_resnet50', pretrained=True)  #, model_dir=model_dir)
    else:
        ofa_network = loaded_model
    ofa_network.set_active_subnet(d=d_list, e=e_list, w=w_list)
    subnet = ofa_network.get_active_subnet(preserve_weight=True).to(device())
    net_shapes = _get_ofa_resnet_subnet_io_shapes(subnet, resolution)
    #assert len(net_shapes) == sum(OFA_RES_STAGE_MIN_N_BLOCKS) + sum(d_list[1:]) + 1, "Len of net_shapes is {}, sum of OFA_RES... is {}, sum of d_list is {}, d_list is {}".format(len(net_shapes, sum(
    assert len(net_shapes) == sum(OFA_RES_STAGE_MIN_N_BLOCKS) + sum(d_list[1:]) + 1, "Len of net_shapes is {}, sum of OFA_RES is {}, sum of d_list is {}, d_list is {}".format(len(net_shapes), sum(OFA_RES_STAGE_MIN_N_BLOCKS), sum(d_list[1:]), d_list)
    return net_shapes
