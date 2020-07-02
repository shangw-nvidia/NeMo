import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo import logging
from nemo.collections.asr.parts.jasper import init_weights


class ConformerEncoderBlock(torch.nn.Module):
    """A single layer of the Conformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        dropout_layer (float): LayerDrop probability
        layer_norm_eps (float): epsilon parameter for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        param_init (str): parameter initialization method
        ffn_bottleneck_dim (int): bottleneck dimension for the light-weight FFN layer

    """

    def __init__(
        self,
        d_model,
        d_ff,
        n_heads,
        kernel_size,
        dropout,
        dropout_att,
        dropout_layer,
        layer_norm_eps,
        ffn_activation,
        param_init,
        device,
        ffn_bottleneck_dim=0,
    ):
        super(ConformerEncoderBlock, self).__init__()

        self._device = device

        self.n_heads = n_heads
        self.fc_factor = 0.5

        # first half position-wise feed-forward
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward1 = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=ffn_activation,
            param_init=param_init,
            layer_norm_eps=layer_norm_eps,
            bottleneck_dim=ffn_bottleneck_dim,
            device=device,
        )

        # conv module
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.conv = ConformerConvBlock(
            d_model, kernel_size, param_init, dropout=dropout, layer_norm_eps=layer_norm_eps, device=self._device
        )

        # self-attention
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = RelativeMultiheadAttentionMechanism(
            kdim=d_model,
            qdim=d_model,
            adim=d_model,
            odim=d_model,
            n_heads=n_heads,
            dropout=dropout_att,
            bias=True,
            param_init=param_init,
        )

        # second half position-wise feed-forward
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.feed_forward2 = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=ffn_activation,
            param_init=param_init,
            layer_norm_eps=layer_norm_eps,
            bottleneck_dim=ffn_bottleneck_dim,
            device=device,
        )

        self.dropout = nn.Dropout(dropout)
        self.dropout_layer = dropout_layer

    def forward(self, xs, xx_mask=None, pos_embs=None, u=None, v=None, pad_mask=None):
        """Conformer encoder layer definition.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
            xx_mask (ByteTensor): `[B, T, T]`
            pos_embs (LongTensor): `[L, 1, d_model]`
            u (FloatTensor): global parameter for relative positinal embedding
            v (FloatTensor): global parameter for relative positinal embedding
        Returns:
            xs (FloatTensor): `[B, T, d_model]`
            xx_aws (FloatTensor): `[B, H, T, T]`

        """
        # if self.dropout_layer > 0 and self.training and random.random() >= self.dropout_layer:
        #    return xs, None

        # first half FFN
        residual = xs
        # xs = self.norm1(xs)
        xs = self.feed_forward1(xs)
        xs = self.fc_factor * xs + residual  # Macaron FFN
        # xs = self.fc_factor * xs + residual  # Macaron FFN

        # self-attention
        residual = xs
        xs = self.norm3(xs)
        # relative positional encoding
        memory = None
        xs, xx_aws = self.self_attn(xs, xs, memory, pos_embs, xx_mask, u, v)
        #xs = xs + residual
        xs = self.dropout(xs) + residual

        # if pad_mask is not None:
        #    xs.masked_fill_(pad_mask, 0.0)

        # conv
        residual = xs
        # xs = self.norm2(xs)
        xs = self.conv(xs)
        #xs = self.dropout(xs) + residual
        xs = xs + residual

        # second half FFN
        residual = xs
        # xs = self.norm4(xs)
        xs = self.feed_forward2(xs)
        xs = self.fc_factor * xs + residual  # Macaron FFN

        # if pad_mask is not None:
        #    xs.masked_fill_(pad_mask, 0.0)

        return xs, xx_aws


class ConformerConvBlock(nn.Module):
    """A single convolution block for the Conformer encoder.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution
        param_init (str): parameter initialization method
    """

    def __init__(self, d_model, kernel_size, param_init, dropout, layer_norm_eps, device):
        super(ConformerConvBlock, self).__init__()

        self._device = device
        self.d_model = d_model
        assert kernel_size % 2 == 1

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model * 2, kernel_size=1, stride=1, padding=0  # for GLU
        )
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            # padding=kernel_size // 2 - 1,
            padding=kernel_size // 2,
            groups=d_model,
        )  # depthwise
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, stride=1, padding=0)

        self.dropout = nn.Dropout(dropout)

        #self.apply(lambda x: init_weights(x, mode=param_init))
        self.to(self._device)

        # changed here
        # if param_init == 'xavier_uniform':
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        logging.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        for layer in [self.pointwise_conv1, self.pointwise_conv2, self.depthwise_conv]:
            for n, p in layer.named_parameters():
                init_with_xavier_uniform(n, p)

    def forward(self, xs, pad_mask=None):
        """Forward pass.
        Args:
            xs (FloatTensor): `[B, T, d_model]`
        Returns:
            xs (FloatTensor): `[B, T, d_model]`
        """
        B, T, d_model = xs.size()
        assert d_model == self.d_model

        xs = self.layer_norm(xs)
        xs = xs.transpose(2, 1).contiguous()  # `[B, C, T]`
        xs = self.pointwise_conv1(xs)  # `[B, 2 * C, T]`
        xs = xs.transpose(2, 1)  # `[B, T, 2 * C]`
        xs = F.glu(xs)  # `[B, T, C]`

        if pad_mask is not None:
            xs.masked_fill_(pad_mask, 0.0)

        xs = xs.transpose(2, 1).contiguous()  # `[B, C, T]`
        xs = self.depthwise_conv(xs)  # `[B, C, T]`

        xs = self.batch_norm(xs)
        xs = self.activation(xs)

        if pad_mask is not None:
            xs.masked_fill_(pad_mask.transpose(2, 1), 0.0)

        xs = self.pointwise_conv2(xs)  # `[B, C, T]`
        xs = self.dropout(xs)
        xs = xs.transpose(2, 1).contiguous()  # `[B, T, C]`
        return xs


class PositionwiseFeedForward(nn.Module):
    """Positionwise fully-connected feed-forward neural network (FFN) layer.
    Args:
        d_model (int): input and output dimension
        d_ff (int): hidden dimension
        dropout (float): dropout probability
        activation (str): non-linear activation function
        param_init (str): parameter initialization method
        bottleneck_dim (int): bottleneck dimension for low-rank FFN
    """

    def __init__(self, d_model, d_ff, dropout, activation, param_init, layer_norm_eps, bottleneck_dim, device):
        super(PositionwiseFeedForward, self).__init__()

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.bottleneck_dim = bottleneck_dim
        if bottleneck_dim > 0:
            self.w_1_e = nn.Linear(d_model, bottleneck_dim)
            self.w_1_d = nn.Linear(bottleneck_dim, d_ff)
            self.w_2_e = nn.Linear(d_ff, bottleneck_dim)
            self.w_2_d = nn.Linear(bottleneck_dim, d_model)
        else:
            self.w_1 = nn.Linear(d_model, d_ff)
            self.w_2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(p=dropout)

        if activation == 'relu':
            self.activation = torch.relu
        elif activation == 'gelu':
            self.activation = lambda x: gelu(x)
        elif activation == 'gelu_accurate':
            self.activation = lambda x: gelu_accurate(x)
        elif activation == 'glu':
            self.activation = LinearGLUBlock(d_ff)
        elif activation == 'swish':
            self.activation = Swish()
        else:
            raise NotImplementedError(activation)
        logging.info('FFN activation: %s' % activation)

        #self.apply(lambda x: init_weights(x, mode=param_init))
        self._device = device
        self.to(self._device)

        # if param_init == 'xavier_uniform':
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        logging.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0.0)
            elif p.dim() == 2:
                nn.init.xavier_uniform_(p)
            else:
                raise ValueError(n)

    def forward(self, xs):
        """Forward computation.
        Args:
            xs (FloatTensor): `[B, T, d_model]`
        Returns:
            xs (FloatTensor): `[B, T, d_model]`
        """
        if self.bottleneck_dim > 0:
            xs = self.w_2_d(self.w_2_e(self.dropout(self.activation(self.w_1_d(self.w_1_e(xs))))))
            return self.dropout(xs)
        else:
            xs = self.norm1(xs)
            xs = self.w_2(self.dropout(self.activation(self.w_1(xs))))
            return self.dropout(xs)


class RelativeMultiheadAttentionMechanism(nn.Module):
    """Relative multi-head attention layer for TransformerXL.
    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        adim: (int) dimension of the attention space
        odim: (int) dimension of output
        n_heads (int): number of heads
        dropout (float): dropout probability for attenion weights
        bias (bool): use bias term in linear layers
        param_init (str): parameter initialization method
    """

    def __init__(self, kdim, qdim, adim, odim, n_heads, dropout, bias, param_init):
        super(RelativeMultiheadAttentionMechanism, self).__init__()

        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(self.d_k)

        # attention dropout applied AFTER the softmax layer
        self.dropout = nn.Dropout(p=dropout)

        self.w_key = nn.Linear(kdim, adim, bias=bias)
        self.w_value = nn.Linear(kdim, adim, bias=bias)
        self.w_query = nn.Linear(qdim, adim, bias=bias)
        self.w_position = nn.Linear(qdim, adim, bias=bias)
        # TODO: fix later
        self.w_out = nn.Linear(adim, odim, bias=bias)

        #self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        if param_init == 'xavier_uniform':
            self.reset_parameters(bias)

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logging.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        # NOTE: see https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_value.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_key.bias, 0.0)
            nn.init.constant_(self.w_value.bias, 0.0)
            nn.init.constant_(self.w_query.bias, 0.0)

        nn.init.xavier_uniform_(self.w_out.weight)
        if bias:
            nn.init.constant_(self.w_out.bias, 0.0)

    def _rel_shift(self, xs):
        """Calculate relative positional attention efficiently.
        Args:
            xs (FloatTensor): `[B, qlen, klen, H]`
        Returns:
            xs_shifted (FloatTensor): `[B, qlen, klen, H]`
        """
        bs, qlen, klen, n_heads = xs.size()
        # `[qlen, klen, B, H]` -> `[B, qlen, klen, H]`
        xs = xs.permute(1, 2, 0, 3).contiguous().view(qlen, klen, bs * n_heads)

        zero_pad = xs.new_zeros((qlen, 1, bs * n_heads))
        xs_shifted = torch.cat([zero_pad, xs], dim=1).view(klen + 1, qlen, bs * n_heads)[1:].view_as(xs)
        return xs_shifted.view(qlen, klen, bs, n_heads).permute(2, 0, 1, 3)

    def forward(self, key, query, memory, pos_embs, mask, u=None, v=None):
        """Forward computation.
        Args:
            key (FloatTensor): `[B, klen, kdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            memory (FloatTensor): `[B, mlen, d_model]`
            mask (ByteTensor): `[B, qlen, klen+mlen]`
            pos_embs (LongTensor): `[qlen, 1, d_model]`
            u (nn.Parameter): `[H, d_k]`
            v (nn.Parameter): `[H, d_k]`
        Returns:
            cv (FloatTensor): `[B, qlen, vdim]`
            aw (FloatTensor): `[B, H, qlen, klen+mlen]`
        """
        bs, qlen = query.size()[:2]
        klen = key.size(1)
        mlen = memory.size(1) if memory is not None and memory.dim() > 1 else 0
        if mlen > 0:
            key = torch.cat([memory, key], dim=1)

        value = self.w_value(key).view(bs, -1, self.n_heads, self.d_k)  # `[B, klen+mlen, H, d_k]`
        key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)  # `[B, klen+mlen, H, d_k]`
        if mask is not None:
            mask = mask.unsqueeze(3).repeat([1, 1, 1, self.n_heads])
            assert mask.size() == (bs, qlen, mlen + klen, self.n_heads), (
                mask.size(),
                (bs, qlen, klen + mlen, self.n_heads),
            )

        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)  # `[B, qlen, H, d_k]`
        pos_embs = self.w_position(pos_embs)
        pos_embs = pos_embs.view(-1, self.n_heads, self.d_k)  # `[qlen, H, d_k]`

        # content-based attention term: (a) + (c)
        if u is not None:
            AC = torch.einsum("bihd,bjhd->bijh", ((query + u[None, None]), key))  # `[B, qlen, klen+mlen, H]`
        else:
            AC = torch.einsum("bihd,bjhd->bijh", (query, key))  # `[B, qlen, klen+mlen, H]`

        # position-based attention term: (b) + (d)
        if v is not None:
            BD = torch.einsum("bihd,jhd->bijh", ((query + v[None, None]), pos_embs))  # `[B, qlen, klen+mlen, H]`
        else:
            BD = torch.einsum("bihd,jhd->bijh", (query, pos_embs))  # `[B, qlen, klen+mlen, H]`

        # Compute positional attention efficiently
        BD = self._rel_shift(BD)

        # the attention is the sum of content-based and position-based attention
        e = (AC + BD) / self.scale  # `[B, qlen, klen+mlen, H]`

        # Compute attention weights
        if mask is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=e.dtype).numpy().dtype).min)
            e = e.masked_fill_(mask == 0, NEG_INF)  # `[B, qlen, klen+mlen, H]`
        aw = torch.softmax(e, dim=2)
        aw = self.dropout(aw)  # `[B, qlen, klen+mlen, H]`
        cv = torch.einsum("bijh,bjhd->bihd", (aw, value))  # `[B, qlen, H, d_k]`
        cv = cv.contiguous().view(bs, -1, self.n_heads * self.d_k)  # `[B, qlen, H * d_k]`
        cv = self.w_out(cv)
        aw = aw.permute(0, 3, 1, 2)  # `[B, H, qlen, klen+mlen]`

        return cv, aw


class LinearGLUBlock(nn.Module):
    """A linear GLU block.
    Args:
        size (int): input and output dimension
    """

    def __init__(self, size):
        super().__init__()

        self.fc = nn.Linear(size, size * 2)

    def forward(self, xs):
        return F.glu(self.fc(xs), dim=-1)


def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))


def gelu(x):
    if hasattr(nn.functional, 'gelu'):
        return nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# def init_with_xavier_uniform(n, p):
#     if p.dim() == 1:
#         nn.init.constant_(p, 0.0)  # bias
#         logging.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.0))
#     elif p.dim() in [2, 3]:
#         nn.init.xavier_uniform_(p)  # linear layer
#         logging.info('Initialize %s with %s' % (n, 'xavier_uniform'))
#     else:
#         raise ValueError(n)


# def init_with_lecun_normal(n, p, param_init):
#     if p.dim() == 1:
#         nn.init.constant_(p, 0.0)  # bias
#         logging.info('Initialize %s with %s' % (n, 'constant'))
#     elif p.dim() == 2:
#         fan_in = p.size(1)
#         nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(fan_in))  # linear weight
#         logging.info('Initialize %s with %s' % (n, 'lecun'))
#     elif p.dim() == 3:
#         fan_in = p.size(1) * p[0][0].numel()
#         nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(fan_in))  # 1d conv weight
#         logging.info('Initialize %s with %s' % (n, 'lecun'))
#     elif p.dim() == 4:
#         fan_in = p.size(1) * p[0][0].numel()
#         nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(fan_in))  # 2d conv weight
#         logging.info('Initialize %s with %s' % (n, 'lecun'))
#     else:
#         raise ValueError(n)


class XLPositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout):
        """Positional embedding for TransformerXL."""
        super().__init__()
        self.d_model = d_model
        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer("inv_freq", inv_freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, positions, device):
        """Forward computation.
        Args:
            positions (LongTensor): `[L]`
        Returns:
            pos_emb (LongTensor): `[L, 1, d_model]`
        """
        if device:
            positions = positions.to(device)
        # outer product
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        pos_emb = self.dropout(pos_emb)
        return pos_emb.unsqueeze(1)


class ConvEncoder(nn.Module):
    """CNN encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        in_channel (int): number of channels of input features
        channels (list): number of channels in CNN blocks
        kernel_sizes (list): size of kernels in CNN blocks
        strides (list): strides in CNN blocks
        poolings (list): size of poolings in CNN blocks
        dropout (float): probability to drop nodes in hidden-hidden connection
        batch_norm (bool): apply batch normalization
        layer_norm (bool): apply layer normalization
        residual (bool): add residual connections
        bottleneck_dim (int): dimension of the bridge layer after the last layer
        param_init (float): model initialization parameter
        layer_norm_eps (float):

    """

    def __init__(
        self,
        input_dim,
        in_channel,
        channels,
        kernel_sizes,
        strides,
        poolings,
        dropout,
        batch_norm,
        layer_norm,
        residual,
        bottleneck_dim,
        param_init,
        device,
        layer_norm_eps=1e-12,
    ):

        super(ConvEncoder, self).__init__()

        (channels, kernel_sizes, strides, poolings), is_1dconv = parse_cnn_config(
            channels, kernel_sizes, strides, poolings
        )

        self._device = device

        self.is_1dconv = is_1dconv
        self.in_channel = in_channel
        assert input_dim % in_channel == 0
        self.input_freq = input_dim // in_channel
        self.residual = residual
        self.bridge = None

        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes) == len(strides) == len(poolings)

        self.layers = nn.ModuleList()
        C_i = input_dim if is_1dconv else in_channel
        in_freq = self.input_freq
        for lth in range(len(channels)):
            if is_1dconv:
                block = Conv1dBlock(
                    in_channel=C_i,
                    out_channel=channels[lth],
                    kernel_size=kernel_sizes[lth],  # T
                    stride=strides[lth],  # T
                    pooling=poolings[lth],  # T
                    dropout=dropout,
                    batch_norm=batch_norm,
                    layer_norm=layer_norm,
                    layer_norm_eps=layer_norm_eps,
                    residual=residual,
                    param_init=param_init,
                    device=self._device,
                )
            else:
                block = Conv2dBlock(
                    input_dim=in_freq,
                    in_channel=C_i,
                    out_channel=channels[lth],
                    kernel_size=kernel_sizes[lth],  # (T,F)
                    stride=strides[lth],  # (T,F)
                    pooling=poolings[lth],  # (T,F)
                    dropout=dropout,
                    batch_norm=batch_norm,
                    layer_norm=layer_norm,
                    layer_norm_eps=layer_norm_eps,
                    residual=residual,
                    param_init=param_init,
                    device=self._device,
                )
            self.layers += [block]
            in_freq = block.output_dim
            C_i = channels[lth]

        self._odim = C_i if is_1dconv else int(C_i * in_freq)

        if bottleneck_dim > 0:
            self.bridge = nn.Linear(self._odim, bottleneck_dim)
            self._odim = bottleneck_dim

        # calculate subsampling factor
        self._factor = 1
        if poolings:
            for p in poolings:
                self._factor *= p if is_1dconv else p[0]

        # changed here
        #param_init = "xavier_uniform"
        self.to(self._device)

        self.output_dim = self._odim
        self.subsampling_factor = 1
        self.apply(lambda x: init_weights(x, mode=param_init))
        #self.reset_parameters(param_init)

    # @staticmethod
    # def add_args(parser, args):
    #     """Add arguments."""
    #     group = parser.add_argument_group("CNN encoder")
    #     group.add_argument('--conv_in_channel', type=int, default=1, help='input dimension of the first CNN block')
    #     group.add_argument(
    #         '--conv_channels', type=str, default="", help='delimited list of channles in each CNN block'
    #     )
    #     group.add_argument(
    #         '--conv_kernel_sizes', type=str, default="", help='delimited list of kernel sizes in each CNN block'
    #     )
    #     group.add_argument('--conv_strides', type=str, default="", help='delimited list of strides in each CNN block')
    #     group.add_argument(
    #         '--conv_poolings', type=str, default="", help='delimited list of poolings in each CNN block'
    #     )
    #     group.add_argument(
    #         '--conv_batch_norm', type=strtobool, default=False, help='apply batch normalization in each CNN block'
    #     )
    #     group.add_argument(
    #         '--conv_layer_norm', type=strtobool, default=False, help='apply layer normalization in each CNN block'
    #     )
    #     group.add_argument(
    #         '--conv_bottleneck_dim',
    #         type=int,
    #         default=0,
    #         help='dimension of the bottleneck layer between CNN and the subsequent RNN/Transformer layers',
    #     )
    #     return parser

    # def reset_parameters(self, param_init):
    #     """Initialize parameters with lecun style."""
    #     logging.info('===== Initialize %s with lecun style =====' % self.__class__.__name__)
    #     for n, p in self.named_parameters():
    #         init_with_lecun_normal(n, p, param_init)

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, F]`
            xlens (list): A list of length `[B]`
        Returns:
            xs (FloatTensor): `[B, T', F']`
            xlens (list): A list of length `[B]`

        """
        B, T, F = xs.size()
        C_i = self.in_channel
        if not self.is_1dconv:
            xs = xs.view(B, T, C_i, F // C_i).contiguous().transpose(2, 1)  # `[B, C_i, T, F // C_i]`

        for block in self.layers:
            xs, xlens = block(xs, xlens)
        if not self.is_1dconv:
            B, C_o, T, F = xs.size()
            xs = xs.transpose(2, 1).contiguous().view(B, T, -1)  # `[B, T', C_o * F']`

        # Bridge layer
        if self.bridge is not None:
            xs = self.bridge(xs)

        return xs, xlens


class Conv1dBlock(torch.nn.Module):
    """1d-CNN block."""

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        pooling,
        dropout,
        batch_norm,
        layer_norm,
        layer_norm_eps,
        residual,
        param_init,
        device
    ):

        super(Conv1dBlock, self).__init__()

        self._device = device

        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.residual = residual
        self.dropout = nn.Dropout(p=dropout)

        # 1st layer
        self.conv1 = nn.Conv1d(
            in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=1
        )
        self._odim = update_lens_1d([in_channel], self.conv1)[0]
        self.batch_norm1 = nn.BatchNorm1d(out_channel) if batch_norm else lambda x: x
        self.layer_norm1 = nn.LayerNorm(out_channel, eps=layer_norm_eps) if layer_norm else lambda x: x

        # 2nd layer
        self.conv2 = nn.Conv1d(
            in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=1
        )
        self._odim = update_lens_1d([self._odim], self.conv2)[0]
        self.batch_norm2 = nn.BatchNorm1d(out_channel) if batch_norm else lambda x: x
        self.layer_norm2 = nn.LayerNorm(out_channel, eps=layer_norm_eps) if layer_norm else lambda x: x

        # Max Pooling
        self.pool = None
        if pooling > 1:
            self.pool = nn.MaxPool1d(kernel_size=pooling, stride=pooling, padding=0, ceil_mode=True)
            # NOTE: If ceil_mode is False, remove last feature when the dimension of features are odd.
            self._odim = update_lens_1d([self._odim], self.pool)[0].item()
            if self._odim % 2 != 0:
                self._odim = (self._odim // 2) * 2
                # TODO(hirofumi0810): more efficient way?

        self.apply(lambda x: init_weights(x, mode=param_init))
        self.to(self._device)

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, F]`
            xlens (IntTensor): `[B]`
        Returns:
            xs (FloatTensor): `[B, T', F']`
            xlens (IntTensor): `[B]`

        """
        residual = xs

        xs = xs.transpose(2, 1)
        xs = self.conv1(xs)
        xs = xs.transpose(2, 1)
        xs = self.batch_norm1(xs)
        xs = self.layer_norm1(xs)
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens_1d(xlens, self.conv1)

        xs = xs.transpose(2, 1)
        xs = self.conv2(xs)
        xs = xs.transpose(2, 1)
        xs = self.batch_norm2(xs)
        xs = self.layer_norm2(xs)
        if self.residual and xs.size() == residual.size():
            xs += residual  # NOTE: this is the same place as in ResNet
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens_1d(xlens, self.conv2)

        if self.pool is not None:
            xs = self.pool(xs)
            xlens = update_lens_1d(xlens, self.pool)

        return xs, xlens


class Conv2dBlock(torch.nn.Module):
    """2d-CNN block."""

    def __init__(
        self,
        input_dim,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        pooling,
        dropout,
        batch_norm,
        layer_norm,
        layer_norm_eps,
        residual,
        param_init,
        device,
    ):

        super(Conv2dBlock, self).__init__()

        self._device = device
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.residual = residual
        self.dropout = nn.Dropout(p=dropout)

        # 1st layer
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=tuple(kernel_size),
            stride=tuple(stride),
            padding=(1, 1),
        )
        self._odim = update_lens_2d([input_dim], self.conv1, dim=1)[0]
        self.batch_norm1 = nn.BatchNorm2d(out_channel) if batch_norm else lambda x: x
        self.layer_norm1 = (
            LayerNorm2D(out_channel * self._odim.item(), eps=layer_norm_eps) if layer_norm else lambda x: x
        )

        # 2nd layer
        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=tuple(kernel_size),
            stride=tuple(stride),
            padding=(1, 1),
        )
        self._odim = update_lens_2d([self._odim], self.conv2, dim=1)[0]
        self.batch_norm2 = nn.BatchNorm2d(out_channel) if batch_norm else lambda x: x
        self.layer_norm2 = (
            LayerNorm2D(out_channel * self._odim.item(), eps=layer_norm_eps) if layer_norm else lambda x: x
        )

        # Max Pooling
        self.pool = None
        if len(pooling) > 0 and np.prod(pooling) > 1:
            self.pool = nn.MaxPool2d(kernel_size=tuple(pooling), stride=tuple(pooling), padding=(0, 0), ceil_mode=True)
            # NOTE: If ceil_mode is False, remove last feature when the dimension of features are odd.
            self._odim = update_lens_2d([self._odim], self.pool, dim=1)[0].item()
            if self._odim % 2 != 0:
                self._odim = (self._odim // 2) * 2
                # TODO(hirofumi0810): more efficient way?

        self.apply(lambda x: init_weights(x, mode=param_init))
        self.to(self._device)

        # changed here
        self.output_dim = self._odim

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, C_i, T, F]`
            xlens (IntTensor): `[B]`
        Returns:
            xs (FloatTensor): `[B, C_o, T', F']`
            xlens (IntTensor): `[B]`

        """
        residual = xs

        xs = self.conv1(xs)
        xs = self.batch_norm1(xs)
        xs = self.layer_norm1(xs)
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens_2d(xlens, self.conv1, dim=0)

        xs = self.conv2(xs)
        xs = self.batch_norm2(xs)
        xs = self.layer_norm2(xs)
        if self.residual and xs.size() == residual.size():
            xs += residual  # NOTE: this is the same place as in ResNet
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens_2d(xlens, self.conv2, dim=0)

        if self.pool is not None:
            xs = self.pool(xs)
            xlens = update_lens_2d(xlens, self.pool, dim=0)

        return xs, xlens


class LayerNorm2D(nn.Module):
    """Layer normalization for CNN outputs."""

    def __init__(self, dim, eps=1e-12):

        super(LayerNorm2D, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, xs):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, C, T, F]`
        Returns:
            xs (FloatTensor): `[B, C, T, F]`

        """
        B, C, T, F = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(B, T, C * F)
        xs = self.norm(xs)
        xs = xs.view(B, T, C, F).transpose(2, 1)
        return xs


def update_lens_1d(seq_lens, layer, device_id=-1):
    """Update lenghts (frequency or time).

    Args:
        seq_lens (list or IntTensor):
        layer (nn.Conv1d or nn.MaxPool1d):
        device_id (int):
    Returns:
        seq_lens (IntTensor):

    """
    if seq_lens is None:
        return seq_lens
    assert type(layer) in [nn.Conv1d, nn.MaxPool1d]
    seq_lens = [_update_1d(seq_len, layer) for seq_len in seq_lens]
    seq_lens = torch.IntTensor(seq_lens)
    if device_id >= 0:
        seq_lens = seq_lens.cuda(device_id)
    return seq_lens


def _update_1d(seq_len, layer):
    if type(layer) == nn.MaxPool1d and layer.ceil_mode:
        return math.ceil((seq_len + 1 + 2 * layer.padding[0] - (layer.kernel_size[0] - 1) - 1) / layer.stride[0] + 1)
    else:
        return math.floor((seq_len + 2 * layer.padding[0] - (layer.kernel_size[0] - 1) - 1) / layer.stride[0] + 1)


def update_lens_2d(seq_lens, layer, dim=0, device_id=-1):
    """Update lenghts (frequency or time).

    Args:
        seq_lens (list or IntTensor):
        layer (nn.Conv2d or nn.MaxPool2d):
        dim (int):
        device_id (int):
    Returns:
        seq_lens (IntTensor):

    """
    if seq_lens is None:
        return seq_lens
    assert type(layer) in [nn.Conv2d, nn.MaxPool2d]
    seq_lens = [_update_2d(seq_len, layer, dim) for seq_len in seq_lens]
    seq_lens = torch.IntTensor(seq_lens)
    if device_id >= 0:
        seq_lens = seq_lens.cuda(device_id)
    return seq_lens


def _update_2d(seq_len, layer, dim):
    if type(layer) == nn.MaxPool2d and layer.ceil_mode:
        return math.ceil(
            (seq_len + 1 + 2 * layer.padding[dim] - (layer.kernel_size[dim] - 1) - 1) / layer.stride[dim] + 1
        )
    else:
        return math.floor(
            (seq_len + 2 * layer.padding[dim] - (layer.kernel_size[dim] - 1) - 1) / layer.stride[dim] + 1
        )


def parse_cnn_config(channels, kernel_sizes, strides, poolings):
    _channels, _kernel_sizes, _strides, _poolings = [], [], [], []
    is_1dconv = '(' not in kernel_sizes
    if len(channels) > 0:
        _channels = [int(c) for c in channels.split('_')]
    if len(kernel_sizes) > 0:
        if is_1dconv:
            _kernel_sizes = [int(c) for c in kernel_sizes.split('_')]
        else:
            _kernel_sizes = [
                [int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                for c in kernel_sizes.split('_')
            ]
    if len(strides) > 0:
        if is_1dconv:
            assert '(' not in _strides and ')' not in _strides
            _strides = [int(s) for s in strides.split('_')]
        else:
            _strides = [
                [int(s.split(',')[0].replace('(', '')), int(s.split(',')[1].replace(')', ''))]
                for s in strides.split('_')
            ]
    if len(poolings) > 0:
        if is_1dconv:
            assert '(' not in poolings and ')' not in poolings
            _poolings = [int(p) for p in strides.split('_')]
        else:
            _poolings = [
                [int(p.split(',')[0].replace('(', '')), int(p.split(',')[1].replace(')', ''))]
                for p in poolings.split('_')
            ]
    return (_channels, _kernel_sizes, _strides, _poolings), is_1dconv


def blockwise(xs, N_l, N_c, N_r):
    bs, xmax, idim = xs.size()

    n_blocks = xmax // N_c
    if xmax % N_c != 0:
        n_blocks += 1
    xs_tmp = xs.new_zeros(bs, n_blocks, N_l + N_c + N_r, idim)
    xs_pad = torch.cat([xs.new_zeros(bs, N_l, idim), xs, xs.new_zeros(bs, N_r, idim)], dim=1)
    for blc_id, t in enumerate(range(N_l, N_l + xmax, N_c)):
        xs_chunk = xs_pad[:, t - N_l : t + (N_c + N_r)]
        xs_tmp[:, blc_id, : xs_chunk.size(1), :] = xs_chunk
    xs = xs_tmp.view(bs * n_blocks, N_l + N_c + N_r, idim)

    return xs


def init_with_xavier_uniform(n, p):
    if p.dim() == 1:
        nn.init.constant_(p, 0.)  # bias
        logging.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.))
    elif p.dim() in [2, 3]:
        nn.init.xavier_uniform_(p)  # linear layer
        logging.info('Initialize %s with %s' % (n, 'xavier_uniform'))
    else:
        raise ValueError(n)