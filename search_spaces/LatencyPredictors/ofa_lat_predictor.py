import os
import torch
from .constants import *
from .utils.model_utils import model_load, device


SAVED_MODELS_DIR = "models/Latency"
CACHE_DIR = "_cache"


"""
Normalization constants
"""
OFA_NORM_CONSTANTS = {
    "ofa_pn_op_graph_npu_lat": 32529.526315789475,
    "ofa_pn_op_graph_gpu_lat": 7.015289268493652,
    "ofa_pn_op_graph_cpu_lat": 0.18810529708862306,

    "ofa_mbv3_op_graph_npu_lat": 29956.36842105263,
    "ofa_mbv3_op_graph_gpu_lat": 11.565698537826538,
    "ofa_mbv3_op_graph_cpu_lat": 0.7969109535217285,

    "ofa_resnet_op_graph_gpu_lat": 7.504615359306335,
    "ofa_resnet_op_graph_cpu_lat": 0.4949644088745117,

    "ofa_pn_op_graph_npu_b_lat": 7713.65,
    "ofa_mbv3_op_graph_npu_b_lat": 5769.15,
}


"""
Predictor output units, after un-normalization
"""
OFA_PREDICTOR_OUTPUT_UNITS = {
    "ofa_pn_op_graph_npu_lat": "us",
    "ofa_pn_op_graph_gpu_lat": "ms",
    "ofa_pn_op_graph_cpu_lat": "s",

    "ofa_mbv3_op_graph_npu_lat": "us",
    "ofa_mbv3_op_graph_gpu_lat": "ms",
    "ofa_mbv3_op_graph_cpu_lat": "s",

    "ofa_resnet_op_graph_gpu_lat": "ms",
    "ofa_resnet_op_graph_cpu_lat": "s",

    "ofa_pn_op_graph_npu_b_lat": "us",
    "ofa_mbv3_op_graph_npu_b_lat": "us",
}


"""
Predict functions
"""
def ofa_op_graph_lat_predict_batch(batch_net_configs, resolutions, model,
                                   norm_constant, batch_size=4096, sub_space="pn"):
    """
    Call this to predict for a batch of networks
    :param batch_net_configs: A list of net_configs
    :param resolutions: A list of either 192, 208, 224
    :param model: Pre-trained model
    :param norm_constant: Normalization constant
    :param sub_space: pn or mbv3
    :return: List of latencies between 0 and 1
    """
    from .model_src.predictor.dataloaders import RNNShapeRegressDataLoader
    from .model_src.search_space.ofa_profile.arch_gen import get_ofa_pn_net_idx_shape_feat, \
        get_ofa_mbv3_net_idx_shape_feat

    def _batch_fwd_func(_model, _batch):
        # Define how a batch is handled by the model
        node_feature_tsr = _batch[DK_BATCH_NODE_FEATURE_TSR]
        shape_tsr = _batch[DK_BATCH_NODE_SHAPE_TSR]
        return _model(node_feature_tsr, shape_tsr)

    # Sub-space select
    if sub_space == "mbv3":
        feat_func = get_ofa_mbv3_net_idx_shape_feat
    elif sub_space == "pn":
        feat_func = get_ofa_pn_net_idx_shape_feat
    else:
        raise ValueError("Unknown space: {}".format(sub_space))

    g_data = []
    for ni, net in enumerate(batch_net_configs):
        resolution = resolutions[ni]
        op_inds, shapes = feat_func(net,
                                    H=resolution, W=resolution,
                                    H_max=224, W_max=224,
                                    log_f=lambda _m: _m)
        g_data.append([op_inds, shapes, ni])
    inst_loader = RNNShapeRegressDataLoader(batch_size, g_data, verbose=False)
    ordered_batch_preds = []
    for batch in inst_loader:
        with torch.no_grad():
            model.eval()
            batch_vals = _batch_fwd_func(model, batch)
        batch_preds = batch_vals.squeeze(1).tolist()
        batch_inds = batch[DK_BATCH_TARGET_TSR].tolist()
        ordered_batch_preds.extend(list(zip(batch_preds, batch_inds)))
    ordered_batch_preds.sort(key=lambda t: t[1])
    ordered_batch_preds = [t[0] for t in ordered_batch_preds]
    assert len(ordered_batch_preds) == len(batch_net_configs)
    if norm_constant is not None:
        ordered_batch_preds = [p * norm_constant for p in ordered_batch_preds]
    return ordered_batch_preds


def ofa_op_graph_lat_predict(net_configs, resolution, model,
                             sub_space="pn"):
    """
    Call this to predict for 1 input network
    :param net_configs: According to the definition of OFA space
    :param resolution: 192, 208, 224
    :param sub_space: pn or mbv3
    :param model: Pre-trained model
    :return: latency between 0 and 1
    """
    from .model_src.predictor.dataloaders import RNNShapeRegressDataLoader
    from .model_src.search_space.ofa_profile.arch_gen import get_ofa_pn_net_idx_shape_feat, \
        get_ofa_mbv3_net_idx_shape_feat

    def _batch_fwd_func(_model, _batch):
        # Define how a batch is handled by the model
        node_feature_tsr = _batch[DK_BATCH_NODE_FEATURE_TSR]
        shape_tsr = _batch[DK_BATCH_NODE_SHAPE_TSR]
        return _model(node_feature_tsr, shape_tsr)

    # Sub-space select
    if sub_space == "mbv3":
        feat_func = get_ofa_mbv3_net_idx_shape_feat
    elif sub_space == "pn":
        feat_func = get_ofa_pn_net_idx_shape_feat
    else:
        raise ValueError("Unknown space: {}".format(sub_space))

    g_data = []
    op_inds, shapes = feat_func(net_configs,
                                H=resolution, W=resolution,
                                H_max=224, W_max=224,
                                log_f=lambda _m: _m)
    g_data.append([op_inds, shapes, 1.0])
    inst_loader = RNNShapeRegressDataLoader(1, g_data, verbose=False)
    preds = []
    for batch in inst_loader:
        with torch.no_grad():
            model.eval()
            batch_vals = _batch_fwd_func(model, batch)
        pred = batch_vals.squeeze(1)
        preds.append(pred.detach().squeeze().item())
    assert len(preds) == 1
    return preds[0]


def ofa_op_graph_lat_predict_v2(k_list, e_list, d_list, resolution, model,
                                sub_space="pn"):
    """
    Call this to predict for 1 input network
    :param k_list: According to the definition of OFA-supernet
    :param e_list: According to the definition of OFA-supernet
    :param d_list: According to the definition of OFA-supernet
    :param resolution: 192, 208, 224
    :param sub_space: pn or mbv3
    :param model: Pre-trained model
    :return: latency between 0 and 1
    """
    from search_spaces.LatencyPredictors.model_src.ofa.my_utils import ofa_subnet_args_to_str_configs

    # Sub-space select
    if sub_space == "mbv3":
        block_prefix = "mbconv3"
        assert len(d_list) == 5, "MBV3 space must have 5 depth values corresponding to 5 stages"
        assert len(k_list) == len(e_list) == 20, "MBV3 space k/e list must have exactly 20 blocks"
        net_configs = ofa_subnet_args_to_str_configs(block_prefix, k_list, e_list, d_list)
    elif sub_space == "pn":
        block_prefix = "mbconv2"
        assert len(d_list) == 6, "PN space must have 6 depth values corresponding to 6 stages"
        assert d_list[-1] == 1, "PN space last stage can have 1 and only 1 block"
        assert len(k_list) == len(e_list) >= 21, "PN space k/e list must have at least 21 blocks"
        net_configs = ofa_subnet_args_to_str_configs(block_prefix, k_list, e_list, d_list)
    else:
        raise ValueError("Unknown space: {}".format(sub_space))

    return ofa_op_graph_lat_predict(net_configs, resolution, model,
                                    sub_space=sub_space)


def ofa_resnet_op_graph_lat_predict_batch(batch_nets, resolutions, model,
                                          norm_constant=None, batch_size=512, supernet=None):
    """
    Call this to predict for a batch of networks
    :param batch_nets: A list of (d_list, e_list, w_list)
    :param resolutions: A list of either 192, 208, 224
    :param model: Pre-trained model
    :param norm_constant:
    :param batch_size:
    :return: latency values, 1 for each net in batch_nets
    """
    #from search_spaces.LatencyPredictors.model_src.ofa.model_zoo import ofa_net
    from ofa.model_zoo import ofa_net
    from search_spaces.LatencyPredictors.model_src.predictor.dataloaders import RNNShapeRegressDataLoader
    from search_spaces.LatencyPredictors.model_src.ofa.my_utils import ofa_res_subnet_args_to_str_configs
    from search_spaces.LatencyPredictors.model_src.search_space.ofa_profile.arch_gen import get_ofa_resnet_io_shapes, \
        ofa_resnet_op2idx

    def _batch_fwd_func(_model, _batch):
        # Define how a batch is handled by the model
        node_feature_tsr = _batch[DK_BATCH_NODE_FEATURE_TSR]
        shape_tsr = _batch[DK_BATCH_NODE_SHAPE_TSR]
        return _model(node_feature_tsr, shape_tsr)

    #model_dir = os.sep.join([SAVED_MODELS_DIR, "ofa_checkpoints"])
    #if not os.path.isdir(model_dir):
    #    os.mkdir(model_dir)
    #ofa_network = ofa_net('ofa_resnet50', pretrained=True) #, model_dir=model_dir)

    g_data = []
    op2idx = ofa_resnet_op2idx()
    for ni, (d_list, e_list, w_list) in enumerate(batch_nets):
        resolution = resolutions[ni]
        net_shapes = get_ofa_resnet_io_shapes(d_list, e_list, w_list, resolution,
                                              loaded_model=supernet)
        net_ops, _ = ofa_res_subnet_args_to_str_configs(d_list, e_list, w_list,
                                                        group_by_stage=False)
        op_inds = [op2idx[op] for op in net_ops]
        assert len(op_inds) == len(net_shapes), "{} vs. {}".format(len(op_inds), len(net_shapes))
        g_data.append([op_inds, net_shapes, ni])
    inst_loader = RNNShapeRegressDataLoader(batch_size, g_data,
                                            verbose=False)
    ordered_batch_preds = []
    for batch in inst_loader:
        with torch.no_grad():
            model.eval()
            batch_vals = _batch_fwd_func(model, batch)
        batch_preds = batch_vals.squeeze(1).tolist()
        inst_inds = batch[DK_BATCH_TARGET_TSR].tolist()
        ordered_batch_preds.extend(list(zip(batch_preds, inst_inds)))
    ordered_batch_preds.sort(key=lambda t: t[1])
    ordered_batch_preds = [t[0] for t in ordered_batch_preds]
    assert len(ordered_batch_preds) == len(batch_nets)
    if norm_constant is not None:
        ordered_batch_preds = [p * norm_constant for p in ordered_batch_preds]
    return ordered_batch_preds


"""
Model loaders
"""
def load_ofa_pn_op_graph_npu_lat_predictor(checkpoint=os.sep.join([SAVED_MODELS_DIR,
                                                                  "ofa_pn_op_graph_npu_lat_predictor_best.pt"]),
                                           add_output_activ=False):
    from search_spaces.LatencyPredictors.model_src.search_space.ofa_profile.constants import PN_OP2IDX
    from search_spaces.LatencyPredictors.model_src.predictor.model_perf_predictor import make_rnn_shape_regressor
    model = make_rnn_shape_regressor(n_unique_labels=len(PN_OP2IDX),
                                     out_embed_size=128,
                                     activ=torch.nn.Sigmoid() if add_output_activ else None,
                                     shape_embed_size=16, n_shape_vals=6,
                                     hidden_size=512, n_layers=1,
                                     dropout_prob=0.5)
    sd = model_load(checkpoint)
    model.load_state_dict(sd[CHKPT_MODEL])
    return model.to(device())


def load_ofa_pn_op_graph_gpu_lat_predictor(checkpoint=os.sep.join([SAVED_MODELS_DIR,
                                                                  "ofa_pn_op_graph_gpu_lat_predictor_best.pt"]),
                                           add_output_activ=False):
    from search_spaces.LatencyPredictors.model_src.search_space.ofa_profile.constants import PN_OP2IDX
    from search_spaces.LatencyPredictors.model_src.predictor.model_perf_predictor import make_rnn_shape_regressor
    model = make_rnn_shape_regressor(n_unique_labels=len(PN_OP2IDX),
                                     out_embed_size=128,
                                     activ=torch.nn.Sigmoid() if add_output_activ else None,
                                     shape_embed_size=16, n_shape_vals=6,
                                     hidden_size=512, n_layers=1,
                                     dropout_prob=0.5)
    sd = model_load(checkpoint)
    model.load_state_dict(sd[CHKPT_MODEL])
    return model.to(device())


def load_ofa_pn_op_graph_cpu_lat_predictor(checkpoint=os.sep.join([SAVED_MODELS_DIR,
                                                                  "ofa_pn_op_graph_cpu_lat_predictor_best.pt"]),
                                           add_output_activ=False):
    from search_spaces.LatencyPredictors.model_src.search_space.ofa_profile.constants import PN_OP2IDX
    from search_spaces.LatencyPredictors.model_src.predictor.model_perf_predictor import make_rnn_shape_regressor
    model = make_rnn_shape_regressor(n_unique_labels=len(PN_OP2IDX),
                                     out_embed_size=128,
                                     activ=torch.nn.Sigmoid() if add_output_activ else None,
                                     shape_embed_size=16, n_shape_vals=6,
                                     hidden_size=512, n_layers=1,
                                     dropout_prob=0.5)
    sd = model_load(checkpoint)
    model.load_state_dict(sd[CHKPT_MODEL])
    return model.to(device())


def load_ofa_mbv3_op_graph_npu_lat_predictor(checkpoint=os.sep.join([SAVED_MODELS_DIR,
                                                                    "ofa_mbv3_op_graph_npu_lat_predictor_best.pt"]),
                                             add_output_activ=False):
    from search_spaces.LatencyPredictors.model_src.search_space.ofa_profile.constants import MBV3_OP2IDX
    from search_spaces.LatencyPredictors.model_src.predictor.model_perf_predictor import make_rnn_shape_regressor
    model = make_rnn_shape_regressor(n_unique_labels=len(MBV3_OP2IDX),
                                     out_embed_size=128,
                                     activ=torch.nn.Sigmoid() if add_output_activ else None,
                                     shape_embed_size=16, n_shape_vals=6,
                                     hidden_size=512, n_layers=1,
                                     dropout_prob=0.2)
    sd = model_load(checkpoint)
    model.load_state_dict(sd[CHKPT_MODEL])
    return model.to(device())


def load_ofa_mbv3_op_graph_gpu_lat_predictor(checkpoint=os.sep.join([SAVED_MODELS_DIR,
                                                                    "ofa_mbv3_op_graph_gpu_lat_predictor_best.pt"]),
                                             add_output_activ=False):
    from search_spaces.LatencyPredictors.model_src.search_space.ofa_profile.constants import MBV3_OP2IDX
    from search_spaces.LatencyPredictors.model_src.predictor.model_perf_predictor import make_rnn_shape_regressor
    model = make_rnn_shape_regressor(n_unique_labels=len(MBV3_OP2IDX),
                                     out_embed_size=128,
                                     activ=torch.nn.Sigmoid() if add_output_activ else None,
                                     shape_embed_size=16, n_shape_vals=6,
                                     hidden_size=512, n_layers=1,
                                     dropout_prob=0.2)
    sd = model_load(checkpoint)
    model.load_state_dict(sd[CHKPT_MODEL])
    return model.to(device())


def load_ofa_mbv3_op_graph_cpu_lat_predictor(checkpoint=os.sep.join([SAVED_MODELS_DIR,
                                                                    "ofa_mbv3_op_graph_cpu_lat_predictor_best.pt"]),
                                             add_output_activ=False):
    from search_spaces.LatencyPredictors.model_src.search_space.ofa_profile.constants import MBV3_OP2IDX
    from search_spaces.LatencyPredictors.model_src.predictor.model_perf_predictor import make_rnn_shape_regressor
    model = make_rnn_shape_regressor(n_unique_labels=len(MBV3_OP2IDX),
                                     out_embed_size=128,
                                     activ=torch.nn.Sigmoid() if add_output_activ else None,
                                     shape_embed_size=16, n_shape_vals=6,
                                     hidden_size=512, n_layers=1,
                                     dropout_prob=0.2)
    sd = model_load(checkpoint)
    model.load_state_dict(sd[CHKPT_MODEL])
    return model.to(device())


def load_ofa_resnet_op_graph_lat_predictor(checkpoint, add_output_activ=True):
    # Mostly for internal usage
    from search_spaces.LatencyPredictors.model_src.search_space.ofa_profile.arch_gen import ofa_resnet_op2idx
    from search_spaces.LatencyPredictors.model_src.predictor.model_perf_predictor import make_rnn_shape_regressor
    model = make_rnn_shape_regressor(n_unique_labels=len(ofa_resnet_op2idx()),
                                     out_embed_size=128,
                                     activ=torch.nn.Sigmoid() if add_output_activ else None,
                                     shape_embed_size=16, n_shape_vals=6,
                                     hidden_size=512, n_layers=1,
                                     dropout_prob=0.2)
    sd = model_load(checkpoint)
    model.load_state_dict(sd[CHKPT_MODEL])
    return model.to(device())