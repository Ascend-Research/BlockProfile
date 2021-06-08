import torch
from search_spaces.LatencyPredictors.utils.model_utils import device
from search_spaces.LatencyPredictors.model_src.model_components import RNNAggregator


class NodeShapeEmbedding(torch.nn.Module):

    def __init__(self, n_unique_labels, out_embed_size, shape_embed_size,
                 n_shape_vals=2):
        super(NodeShapeEmbedding, self).__init__()
        self.n_unique_labels = n_unique_labels
        self.n_shape_vals = n_shape_vals
        op_embed_size = out_embed_size - shape_embed_size
        self.op_embed_layer = torch.nn.Embedding(n_unique_labels, op_embed_size)
        self.shape_embed_layer = torch.nn.Linear(n_shape_vals, shape_embed_size)

    def forward(self, node_inds, shape_vals):
        op_embedding = self.op_embed_layer(node_inds.to(device()))
        shape_embedding = self.shape_embed_layer(shape_vals.to(device()))
        node_embedding = torch.cat([op_embedding, shape_embedding], dim=-1)
        return node_embedding


class RNNShapeRegressor(torch.nn.Module):

    def __init__(self, embed_layer, encoder, hidden_size, aggregator,
                 dropout_prob=0., activ=None):
        super(RNNShapeRegressor, self).__init__()
        self.embed_layer = embed_layer
        self.encoder = encoder
        self.aggregator = aggregator
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.regressor = torch.nn.Linear(hidden_size, 1)
        self.activ = activ

    def forward(self, batch_node_tsr, batch_shapes, index=None):
        node_embedding = self.embed_layer(batch_node_tsr, batch_shapes)
        node_embedding = self.encoder(node_embedding)
        batch_embedding = self.aggregator(node_embedding, index=index)
        batch_embedding = self.dropout(batch_embedding)
        out = self.regressor(batch_embedding)
        if self.activ is not None:
            out = self.activ(out)
        return out


def make_rnn_shape_regressor(n_unique_labels, out_embed_size,
                             shape_embed_size, n_shape_vals, hidden_size,
                             activ=None, n_layers=1, dropout_prob=0.0,
                             aggr_method="squeeze"):
    from search_spaces.LatencyPredictors.model_src.model_components import SimpleRNN
    embed_layer = NodeShapeEmbedding(n_unique_labels, out_embed_size, shape_embed_size, n_shape_vals)
    if aggr_method == "squeeze":
        encoder = SimpleRNN(out_embed_size, hidden_size,
                            n_layers=n_layers, dropout_prob=dropout_prob,
                            return_output_vector_only=True,
                            return_aggr_vector_only=True)
    else:
        encoder = SimpleRNN(out_embed_size, hidden_size,
                            n_layers=n_layers, dropout_prob=dropout_prob,
                            return_output_vector_only=True,
                            return_aggr_vector_only=False)
    aggregator = RNNAggregator(aggr_method=aggr_method)
    regressor = RNNShapeRegressor(embed_layer, encoder, 2 * hidden_size, aggregator,
                                  dropout_prob=dropout_prob, activ=activ)
    return regressor
