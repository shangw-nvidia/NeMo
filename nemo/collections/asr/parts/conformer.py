import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo import logging
from nemo.collections.asr.parts.jasper import init_weights

from torch.nn import LayerNorm


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
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
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
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.conv = ConformerConvBlock(
            d_model, kernel_size, param_init, dropout=dropout, layer_norm_eps=layer_norm_eps, device=self._device
        )

        # self-attention
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)

        self.self_attn = RelPositionMultiHeadedAttention(n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att)
        # self.self_attn = RelativeMultiheadAttentionMechanism(
        #     kdim=d_model,
        #     qdim=d_model,
        #     adim=d_model,
        #     odim=d_model,
        #     n_heads=n_heads,
        #     dropout=dropout_att,
        #     bias=True,
        #     param_init=param_init,
        # )

        # second half position-wise feed-forward
        self.norm4 = LayerNorm(d_model, eps=layer_norm_eps)

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

        self.norm5 = LayerNorm(d_model, eps=layer_norm_eps)

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
        if self.dropout_layer > 0 and self.training and random.random() < self.dropout_layer:
            return xs

        # first half FFN
        residual = xs
        xs = self.norm1(xs)
        xs = self.feed_forward1(xs)
        # xs = self.fc_factor * xs + residual  # Macaron FFN
        xs = self.fc_factor * self.dropout(xs) + residual  # Macaron FFN

        # if pad_mask is not None:
        #    xs.masked_fill_(pad_mask, 0.0)

        # self-attention
        residual = xs
        xs = self.norm3(xs)
        # relative positional encoding
        memory = None
        # if pos_embs is not None:

        xs = self.self_attn(query=xs, key=xs, value=xs, pos_emb=pos_embs, mask=xx_mask)

        # else:
        #     x_att = self.self_attn(xs, xs, xs, xx_mask)

        #xs, xx_aws = self.self_attn(xs, xs, memory, pos_embs, xx_mask, u, v)
        # xs = xs + residual
        xs = self.dropout(xs) + residual

        # conv
        residual = xs
        xs = self.norm2(xs)
        # if pad_mask is not None:
        #     xs.masked_fill_(pad_mask, 0.0)
        xs = self.conv(xs)
        xs = self.dropout(xs) + residual
        # xs = xs + residual

        # second half FFN
        residual = xs
        xs = self.norm4(xs)
        xs = self.feed_forward2(xs)
        # xs = self.fc_factor * xs + residual  # Macaron FFN
        xs = self.fc_factor * self.dropout(xs) + residual  # Macaron FFN

        # if pad_mask is not None:
        #    xs.masked_fill_(pad_mask, 0.0)
        xs = self.norm5(xs)
        return xs


class ConformerConvBlock(nn.Module):
    """A single convolution block for the Conformer encoder.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution
        param_init (str): parameter initialization method
    """

    def __init__(self, d_model, kernel_size, param_init, dropout, layer_norm_eps, device, bias=True):
        super(ConformerConvBlock, self).__init__()

        self._device = device
        self.d_model = d_model
        assert (kernel_size - 1) % 2 == 0

        # self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model * 2, kernel_size=1, stride=1, padding=0, bias=bias  # for GLU
        )
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            # padding=kernel_size // 2 - 1,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
            bias=bias
        )  # depthwise
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, stride=1, padding=0)

        self.dropout = nn.Dropout(dropout)

        # self.apply(lambda x: init_weights(x, mode=param_init))
        # self.to(self._device)

        # changed here
        # if param_init == 'xavier_uniform':
        # self.reset_parameters()

    # def reset_parameters(self):
    #     """Initialize parameters with Xavier uniform distribution."""
    #     logging.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
    #     for layer in [self.pointwise_conv1, self.pointwise_conv2, self.depthwise_conv]:
    #         for n, p in layer.named_parameters():
    #             init_with_xavier_uniform(n, p)

    def forward(self, xs, pad_mask=None):
        """Forward pass.
        Args:
            xs (FloatTensor): `[B, T, d_model]`
        Returns:
            xs (FloatTensor): `[B, T, d_model]`
        """
        B, T, d_model = xs.size()
        assert d_model == self.d_model

        # xs = self.layer_norm(xs)
        #xs = xs.transpose(2, 1).contiguous()  # `[B, C, T]`
        xs = xs.transpose(2, 1)  # `[B, C, T]`
        xs = self.pointwise_conv1(xs)  # `[B, 2 * C, T]`
        #xs = xs.transpose(2, 1)  # `[B, T, 2 * C]`
        xs = nn.functional.glu(xs, dim=1)  # `[B, T, C]`
        #
        # if pad_mask is not None:
        #     xs.masked_fill_(pad_mask, 0.0)

        #xs = xs.transpose(2, 1).contiguous()  # `[B, C, T]`
        xs = self.depthwise_conv(xs)  # `[B, C, T]`

        xs = self.batch_norm(xs)
        xs = self.activation(xs)

        # if pad_mask is not None:
        #     xs.masked_fill_(pad_mask.transpose(2, 1), 0.0)

        xs = self.pointwise_conv2(xs)  # `[B, C, T]`
        # xs = self.dropout(xs)
        xs = xs.transpose(2, 1)  # `[B, T, C]`
        #xs = xs.transpose(2, 1).contiguous()  # `[B, T, C]`
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

        # self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)

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

        # self.apply(lambda x: init_weights(x, mode=param_init))
        self._device = device
        # self.to(self._device)

        # if param_init == 'xavier_uniform':
        #self.reset_parameters()

    # def reset_parameters(self):
    #     """Initialize parameters with Xavier uniform distribution."""
    #     logging.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
    #     for n, p in self.named_parameters():
    #         init_with_xavier_uniform(n, p)

    def forward(self, xs):
        """Forward computation.
        Args:
            xs (FloatTensor): `[B, T, d_model]`
        Returns:
            xs (FloatTensor): `[B, T, d_model]`
        """
        # xs = self.norm1(xs)
        if self.bottleneck_dim > 0:
            xs = self.w_2_d(self.w_2_e(self.dropout(self.activation(self.w_1_d(self.w_1_e(xs))))))
        else:
            xs = self.w_2(self.dropout(self.activation(self.w_1(xs))))
        # return self.dropout(xs)
        return xs


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

        # self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

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

#
# class XLPositionalEmbedding(nn.Module):
#     def __init__(self, d_model, dropout, device=None):
#         """Positional embedding for TransformerXL."""
#         super().__init__()
#         self.d_model = d_model
#         inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
#         self.register_buffer("inv_freq", inv_freq)
#
#         self.dropout = nn.Dropout(p=dropout)
#
#         self._device = device
#
#     def forward(self, positions):
#         """Forward computation.
#         Args:
#             positions (LongTensor): `[L]`
#         Returns:
#             pos_emb (LongTensor): `[L, 1, d_model]`
#         """
#         if self._device:
#             positions = positions.to(self._device)
#         # outer product
#         sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
#         pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
#         pos_emb = self.dropout(pos_emb)
#         return pos_emb.unsqueeze(1)


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
            # block = nn.ReLU(nn.Conv2d(C_i, channels[lth], kernel_size=kernel_sizes[lth], stride=strides[lth]).to(self._device))
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
            )
            self.layers += [block]
            in_freq = block._odim
            C_i = channels[lth]

        # check here
        self._odim = C_i if is_1dconv else int(C_i * in_freq)
        # self._odim = C_i if is_1dconv else int(C_i * 32)

        if bottleneck_dim > 0 and bottleneck_dim != self._odim:
            self.bridge = nn.Linear(self._odim, bottleneck_dim)
            self._odim = bottleneck_dim

        # calculate subsampling factor
        self._factor = 1
        if poolings:
            for p in poolings:
                self._factor *= p if is_1dconv else p[0]

        # changed here
        # param_init = "xavier_uniform"

        self.output_dim = self._odim
        # self.subsampling_factor = 1
        # self.apply(lambda x: init_weights(x, mode=param_init))
        self.reset_parameters(param_init)
        # self.to(self._device)

    def reset_parameters(self, param_init):
        """Initialize parameters with lecun style."""
        logging.info('===== Initialize %s with lecun style =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_lecun_normal(n, p)

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


# class Conv2dBlock(nn.Module):
#     """2d-CNN block."""
#
#     def __init__(
#         self,
#         input_dim,
#         in_channel,
#         out_channel,
#         kernel_size,
#         stride,
#         pooling,
#         dropout,
#         batch_norm,
#         layer_norm,
#         layer_norm_eps,
#         residual,
#     ):
#
#         super(Conv2dBlock, self).__init__()
#
#         self.batch_norm = batch_norm
#         self.layer_norm = layer_norm
#         self.residual = residual
#         self.dropout = nn.Dropout(p=dropout)
#
#         # 1st layer
#         self.conv1 = nn.Conv2d(
#             in_channels=in_channel,
#             out_channels=out_channel,
#             kernel_size=tuple(kernel_size),
#             stride=tuple(stride),
#             padding=(1, 1),
#         )
#         self._odim = update_lens_2d([input_dim], self.conv1, dim=1)[0].item()
#         self.batch_norm1 = nn.BatchNorm2d(out_channel) if batch_norm else lambda x: x
#         self.layer_norm1 = LayerNorm2D(out_channel, self._odim, eps=layer_norm_eps) if layer_norm else lambda x: x
#
#         # 2nd layer
#         self.conv2 = nn.Conv2d(
#             in_channels=out_channel,
#             out_channels=out_channel,
#             kernel_size=tuple(kernel_size),
#             stride=tuple(stride),
#             padding=(1, 1),
#         )
#         self._odim = update_lens_2d([self._odim], self.conv2, dim=1)[0].item()
#         self.batch_norm2 = nn.BatchNorm2d(out_channel) if batch_norm else lambda x: x
#         self.layer_norm2 = LayerNorm2D(out_channel, self._odim, eps=layer_norm_eps) if layer_norm else lambda x: x
#
#         # Max Pooling
#         self.pool = None
#         self._factor = 1
#         if len(pooling) > 0 and np.prod(pooling) > 1:
#             self.pool = nn.MaxPool2d(kernel_size=tuple(pooling), stride=tuple(pooling), padding=(0, 0), ceil_mode=True)
#             # NOTE: If ceil_mode is False, remove last feature when the dimension of features are odd.
#             self._odim = update_lens_2d([self._odim], self.pool, dim=1)[0].item()
#             if self._odim % 2 != 0:
#                 self._odim = (self._odim // 2) * 2
#                 # TODO(hirofumi0810): more efficient way?
#
#             # calculate subsampling factor
#             self._factor *= pooling[0]
#
#     def forward(self, xs, xlens, lookback=False, lookahead=False):
#         """Forward computation.
#         Args:
#             xs (FloatTensor): `[B, C_i, T, F]`
#             xlens (IntTensor): `[B]`
#             lookback (bool): truncate the leftmost frames
#                 because of lookback frames for context
#             lookahead (bool): truncate the rightmost frames
#                 because of lookahead frames for context
#         Returns:
#             xs (FloatTensor): `[B, C_o, T', F']`
#             xlens (IntTensor): `[B]`
#         """
#         residual = xs
#
#         xs = self.conv1(xs)
#         xs = self.batch_norm1(xs)
#         xs = self.layer_norm1(xs)
#         xs = torch.relu(xs)
#         xs = self.dropout(xs)
#         xlens = update_lens_2d(xlens, self.conv1, dim=0)
#         if lookback and xs.size(2) > self.conv1.stride[0]:
#             xs = xs[:, :, self.conv1.stride[0] :]
#         if lookahead and xs.size(2) > self.conv1.stride[0]:
#             xs = xs[:, :, : xs.size(2) - self.conv1.stride[0]]
#
#         xs = self.conv2(xs)
#         xs = self.batch_norm2(xs)
#         xs = self.layer_norm2(xs)
#         if self.residual and xs.size() == residual.size():
#             xs += residual  # NOTE: this is the same place as in ResNet
#         xs = torch.relu(xs)
#         xs = self.dropout(xs)
#         xlens = update_lens_2d(xlens, self.conv2, dim=0)
#         if lookback and xs.size(2) > self.conv2.stride[0]:
#             xs = xs[:, :, self.conv2.stride[0] :]
#         if lookahead and xs.size(2) > self.conv2.stride[0]:
#             xs = xs[:, :, : xs.size(2) - self.conv2.stride[0]]
#
#         if self.pool is not None:
#             xs = self.pool(xs)
#             xlens = update_lens_2d(xlens, self.pool, dim=0)
#
#         return xs, xlens
#
#
# class LayerNorm2D(nn.Module):
#     """Layer normalization for CNN outputs."""
#
#     def __init__(self, channel, idim, eps=1e-12):
#
#         super(LayerNorm2D, self).__init__()
#         self.norm = LayerNorm([channel, idim], eps=eps)
#
#     def forward(self, xs):
#         """Forward computation.
#         Args:
#             xs (FloatTensor): `[B, C, T, F]`
#         Returns:
#             xs (FloatTensor): `[B, C, T, F]`
#         """
#         B, C, T, F = xs.size()
#         xs = xs.transpose(2, 1).contiguous()
#         xs = self.norm(xs)
#         xs = xs.transpose(2, 1)
#         return xs
#
#
# def update_lens_2d(seq_lens, layer, dim=0, device_id=-1):
#     """Update lenghts (frequency or time).
#     Args:
#         seq_lens (list or IntTensor):
#         layer (nn.Conv2d or nn.MaxPool2d):
#         dim (int):
#         device_id (int):
#     Returns:
#         seq_lens (IntTensor):
#     """
#     if seq_lens is None:
#         return seq_lens
#     assert type(layer) in [nn.Conv2d, nn.MaxPool2d]
#     seq_lens = [_update_2d(seq_len, layer, dim) for seq_len in seq_lens]
#     seq_lens = torch.IntTensor(seq_lens)
#     if device_id >= 0:
#         seq_lens = seq_lens.cuda(device_id)
#     return seq_lens
#
#
# def _update_2d(seq_len, layer, dim):
#     if type(layer) == nn.MaxPool2d and layer.ceil_mode:
#         return math.ceil(
#             (seq_len + 1 + 2 * layer.padding[dim] - (layer.kernel_size[dim] - 1) - 1) / layer.stride[dim] + 1
#         )
#     else:
#         return math.floor(
#             (seq_len + 2 * layer.padding[dim] - (layer.kernel_size[dim] - 1) - 1) / layer.stride[dim] + 1
#         )
#
#
# def parse_cnn_config(channels, kernel_sizes, strides, poolings):
#     _channels, _kernel_sizes, _strides, _poolings = [], [], [], []
#     is_1dconv = '(' not in kernel_sizes
#     if len(channels) > 0:
#         _channels = [int(c) for c in channels.split('_')]
#     if len(kernel_sizes) > 0:
#         if is_1dconv:
#             _kernel_sizes = [int(c) for c in kernel_sizes.split('_')]
#         else:
#             _kernel_sizes = [
#                 [int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
#                 for c in kernel_sizes.split('_')
#             ]
#     if len(strides) > 0:
#         if is_1dconv:
#             assert '(' not in _strides and ')' not in _strides
#             _strides = [int(s) for s in strides.split('_')]
#         else:
#             _strides = [
#                 [int(s.split(',')[0].replace('(', '')), int(s.split(',')[1].replace(')', ''))]
#                 for s in strides.split('_')
#             ]
#     if len(poolings) > 0:
#         if is_1dconv:
#             assert '(' not in poolings and ')' not in poolings
#             _poolings = [int(p) for p in poolings.split('_')]
#         else:
#             _poolings = [
#                 [int(p.split(',')[0].replace('(', '')), int(p.split(',')[1].replace(')', ''))]
#                 for p in poolings.split('_')
#             ]
#     return (_channels, _kernel_sizes, _strides, _poolings), is_1dconv


# def init_with_xavier_uniform(n, p):
#     if p.dim() == 1:
#         nn.init.constant_(p, 0.0)  # bias
#         logging.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.0))
#     elif p.dim() in [2, 3, 4]:
#         nn.init.xavier_uniform_(p)  # linear layer
#         logging.info('Initialize %s with %s' % (n, 'xavier_uniform'))
#     else:
#         raise ValueError(n)
#
#
# def init_with_lecun_normal(n, p):
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


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    :param int idim: input dim
    :param int odim: output dim
    :param str activation: activation functions
    :param flaot dropout_rate: dropout rate

    """

    def __init__(self, idim, odim, dropout_rate, activation=nn.ReLU(), subsampling="conformer"):
        super(Conv2dSubsampling, self).__init__()

        self._subsampling = subsampling

        if subsampling == "vggnet":
            self._padding = 0
            self._stride = 2
            self._kernel_size = 2
            self._ceil_mode = True
            conv_channels = 64

            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=3, stride=1, padding=1),
                activation,
                torch.nn.Conv2d(
                    in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                ),
                activation,
                torch.nn.MaxPool2d(
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._padding,
                    ceil_mode=self._ceil_mode,
                ),
                torch.nn.Conv2d(
                    in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                ),
                activation,
                torch.nn.Conv2d(
                    in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                ),
                activation,
                torch.nn.MaxPool2d(
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._padding,
                    ceil_mode=self._ceil_mode,
                ),
            )
        elif subsampling == "vggnet2x":
            self._padding = 0
            self._stride = 2
            self._kernel_size = 2
            self._ceil_mode = True
            conv_channels = 64

            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=3, stride=1, padding=1),
                activation,
                torch.nn.Conv2d(
                    in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                ),
                activation,
                torch.nn.MaxPool2d(
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._padding,
                    ceil_mode=self._ceil_mode,
                ),
            )
        elif subsampling == "conformer":
            self._padding = 0
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False
            conv_channels = odim
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=conv_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._padding,
                ),
                activation,
                torch.nn.Conv2d(
                    in_channels=conv_channels,
                    out_channels=conv_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._padding,
                ),
                activation,
            )
        elif subsampling == "conformer2x":
            self._padding = 0
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False
            conv_channels = odim
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=conv_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._padding,
                ),
                activation,
            )

        out_length = calc_length(
            length=idim,
            padding=self._padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
        )
        if "2x" not in subsampling:
            out_length = calc_length(
                length=out_length,
                padding=self._padding,
                kernel_size=self._kernel_size,
                stride=self._stride,
                ceil_mode=self._ceil_mode,
            )

        # if out_length % 2 != 0:
        #     out_length = (out_length // 2) * 2
        self.out = torch.nn.Linear(conv_channels * out_length, odim)
        # self.out = torch.nn.Linear(conv_channels * (((idim - 1) // 2 - 1) // 2), odim)
        # self.out = torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)

        # self.out = torch.nn.Sequential(
        #     torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
        #     PositionalEncoding(odim, dropout_rate, rel_pos=rel_pos),
        # )

        #self.reset_parameters()

    # def reset_parameters(self):
    #     """Initialize parameters with lecun style."""
    #     logging.info('===== Initialize %s with lecun style =====' % self.__class__.__name__)
    #     for n, p in self.named_parameters():
    #         # init_with_lecun_normal(n, p)
    #         init_with_xavier_uniform(n, p)

    # def forward(self, x, x_mask):
    def forward(self, x, lengths):
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor or Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))

        lengths = [
            calc_length(
                length=length,
                padding=self._padding,
                kernel_size=self._kernel_size,
                stride=self._stride,
                ceil_mode=self._ceil_mode,
            )
            for length in lengths
        ]

        if "2x" not in self._subsampling:
            lengths = [
                calc_length(
                    length=length,
                    padding=self._padding,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    ceil_mode=self._ceil_mode,
                )
                for length in lengths
            ]

        lengths = torch.IntTensor(lengths).to(x.device)

        # if x_mask is None:
        #    return x, None
        # return x, x_mask[:, :, :-2:2][:, :, :-2:2]
        return x, lengths


def calc_length(length, padding, kernel_size, stride, ceil_mode):
    if ceil_mode:
        length = math.ceil((length + (2 * padding) - (kernel_size - 1) - 1) / float(stride) + 1)
    else:
        length = math.floor((length + (2 * padding) - (kernel_size - 1) - 1) / float(stride) + 1)
    return length


# class Conv2dSubsampling(nn.Module):
#     """Convolutional 2D subsampling (to 1/4 length)
#     :param int idim: input dim
#     :param int odim: output dim
#     :param flaot dropout_rate: dropout rate
#     """
#
#     def __init__(self, idim, odim, dropout_rate):
#         super(Conv2dSubsampling, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, odim, 3, 2),
#             nn.ReLU(),
#             nn.Conv2d(odim, odim, 3, 2),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         """Subsample x
#         :param torch.Tensor x: input tensor
#         :param torch.Tensor x_mask: input mask
#         :return: subsampled x and mask
#         :rtype Tuple[torch.Tensor, torch.Tensor]
#         """
#         x = x.unsqueeze(1)  # (b, c, t, f)
#         x = self.conv(x)
#         b, c, t, f = x.size()
#         x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
#         return x



class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.
        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :return torch.Tensor transformed query, key and value
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor scores: (batch, time1, time2)
        :param torch.Tensor mask: (batch, time1, time2)
        :return torch.Tensor transformed `value` (batch, time2, d_model)
            weighted by the attention score (batch, time1, time2)
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(
                np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'.
        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)



class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional ecoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu=False):
        """Compute relative positinal encoding.
        :param torch.Tensor x: (batch, time, size)
        :param bool zero_triu: return the lower triangular part of the matrix
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor pos_emb: (batch, time1, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)