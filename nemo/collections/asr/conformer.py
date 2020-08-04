# Copyright (c) 2019 NVIDIA Corporation
import copy
import math
from typing import Optional

import torch
import torch.nn as nn

from .parts.conformer import ConformerEncoderBlock, Conv2dSubsampling#, ConvEncoder #, XLPositionalEmbedding
from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import *
from nemo.utils import logging
from nemo.utils.decorators import add_port_docs

from torch.nn import LayerNorm


class ConformerEncoder(TrainableNM):
    """
    Jasper Encoder creates the pre-processing (prologue), Jasper convolution
    block, and the first 3 post-processing (epilogue) layers as described in
    Jasper (https://arxiv.org/abs/1904.03288)

    Args:
        jasper (list): A list of dictionaries. Each element in the list
            represents the configuration of one Jasper Block. Each element
            should contain::

                {
                    # Required parameters
                    'filters' (int) # Number of output channels,
                    'repeat' (int) # Number of sub-blocks,
                    'kernel' (int) # Size of conv kernel,
                    'stride' (int) # Conv stride
                    'dilation' (int) # Conv dilation
                    'dropout' (float) # Dropout probability
                    'residual' (bool) # Whether to use residual or not.
                    # Optional parameters
                    'residual_dense' (bool) # Whether to use Dense Residuals
                        # or not. 'residual' must be True for 'residual_dense'
                        # to be enabled.
                        # Defaults to False.
                    'separable' (bool) # Whether to use separable convolutions.
                        # Defaults to False
                    'groups' (int) # Number of groups in each conv layer.
                        # Defaults to 1
                    'heads' (int) # Sharing of separable filters
                        # Defaults to -1
                    'tied' (bool)  # Whether to use the same weights for all
                        # sub-blocks.
                        # Defaults to False
                    'se' (bool)  # Whether to add Squeeze and Excitation
                        # sub-blocks.
                        # Defaults to False
                    'se_reduction_ratio' (int)  # The reduction ratio of the Squeeze
                        # sub-module.
                        # Must be an integer > 1.
                        # Defaults to 8.
                    'se_context_window' (int) # The size of the temporal context
                        # provided to SE sub-module.
                        # Must be an integer. If value <= 0, will perform global
                        # temporal pooling (global context).
                        # If value >= 1, will perform stride 1 average pooling to
                        # compute context window.
                    'se_interpolation_mode' (str) # Interpolation mode of timestep dimension.
                        # Used only if context window is > 1.
                        # The modes available for resizing are: `nearest`, `linear` (3D-only),
                        # `bilinear`, `area`
                    'kernel_size_factor' (float)  # Conv kernel size multiplier
                        # Can be either an int or float
                        # Kernel size is recomputed as below:
                        # new_kernel_size = int(max(1, (kernel_size * kernel_width)))
                        # to prevent kernel sizes than 1.
                        # Note: If rescaled kernel size is an even integer,
                        # adds 1 to the rescaled kernel size to allow "same"
                        # padding.
                    'stride_last' (bool) # Bool flag to determine whether each
                        # of the the repeated sub-blockss will perform a stride,
                        # or only the last sub-block will perform a strided convolution.
                }

        activation (str): Activation function used for each sub-blocks. Can be
            one of ["hardtanh", "relu", "selu", "swish"].
        feat_in (int): Number of channels being input to this module
        normalization_mode (str): Normalization to be used in each sub-block.
            Can be one of ["batch", "layer", "instance", "group"]
            Defaults to "batch".
        residual_mode (str): Type of residual connection.
            Can be "add", "stride_add" or "max".
            "stride_add" mode performs strided convolution prior to residual
            addition.
            Defaults to "add".
        norm_groups (int): Number of groups for "group" normalization type.
            If set to -1, number of channels is used.
            Defaults to -1.
        conv_mask (bool): Controls the use of sequence length masking prior
            to convolutions.
            Defaults to True.
        frame_splicing (int): Defaults to 1.
        init_mode (str): Describes how neural network parameters are
            initialized. Options are ['xavier_uniform', 'xavier_normal',
            'kaiming_uniform','kaiming_normal'].
            Defaults to "xavier_uniform".
    """

    length: Optional[torch.Tensor]

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "audio_signal": NeuralType(
            #    {0: AxisType(BatchTag), 1: AxisType(SpectrogramSignalTag), 2: AxisType(ProcessedTimeTag),}
            # ),
            # "length": NeuralType({0: AxisType(BatchTag)}),
            "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "outputs": NeuralType(
            #    {0: AxisType(BatchTag), 1: AxisType(EncodedRepresentationTag), 2: AxisType(ProcessedTimeTag),}
            # ),
            # "encoded_lengths": NeuralType({0: AxisType(BatchTag)}),
            "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def _disabled_deployment_input_ports(self):
        return set(["length"])

    @property
    def _disabled_deployment_output_ports(self):
        return set(["encoded_lengths"])

    # def _prepare_for_deployment(self):
    #     m_count = 0
    #     for m in self.modules():
    #         if type(m).__name__ == "MaskedConv1d":
    #             m.use_mask = False
    #             m_count += 1
    #     logging.warning(f"Turned off {m_count} masked convolutions")
    #
    #     input_example = torch.randn(16, self.__feat_in, 256)
    #     return input_example, None

    # self,
    # jasper,
    # activation,
    # feat_in,
    # normalization_mode = "batch",
    # residual_mode = "add",
    # norm_groups = -1,
    # conv_mask = True,
    # frame_splicing = 1,
    # init_mode = 'xavier_uniform',
    #
    def __init__(
        self,
        feat_in,
        lstm_hidden_size,
        n_layers,
        d_model,
        n_heads,
        pe_type,
        conv_channels,
        input_dim,
        conv_param_init,
        conv_in_channel,
        conv_kernel_sizes,
        conv_strides,
        conv_layer_norm,
        conv_batch_norm,
        layer_norm_eps,
        conv_poolings,
        dropout,
        ff_expansion_factor,
        kernel_size,
        dropout_att,
        dropout_layer,
        ffn_activation,
        param_init,
        ffn_bottleneck_dim,
        last_proj_dim,
        dropout_in,
        subsampling,
    ):
        super().__init__()

        d_ff = d_model * ff_expansion_factor

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pe_type = pe_type

        # for bridge layers
        self.bridge = None

        self.scale = math.sqrt(self.d_model)

        # Setting for CNNs
        if subsampling:
            self.conv = Conv2dSubsampling(
                idim=input_dim, odim=d_model, dropout_rate=0.0, activation=nn.ReLU(), subsampling=subsampling
            )
            self._odim = d_model
            #
            # self.conv = ConvEncoder(
            #     input_dim,
            #     in_channel=conv_in_channel,
            #     channels=conv_channels,
            #     kernel_sizes=conv_kernel_sizes,
            #     strides=conv_strides,
            #     poolings=conv_poolings,
            #     dropout=0.0,
            #     batch_norm=conv_batch_norm,
            #     layer_norm=conv_layer_norm,
            #     layer_norm_eps=layer_norm_eps,
            #     residual=False,
            #     bottleneck_dim=d_model,
            #     param_init=conv_param_init,
            #     device=self._device,
            # )
            # self._odim = self.conv.output_dim
        else:
            self.conv = None
            self._odim = input_dim  # * n_splices * n_stacks
            self.embed = nn.Linear(self._odim, d_model)

        # calculate subsampling factor
        # self._factor = 1
        # if self.conv is not None:
        #     self._factor *= self.conv.subsampling_factor

        self.pos_enc = RelPositionalEncoding(d_model=d_model, dropout_rate=dropout)

        # self.pos_emb = XLPositionalEmbedding(
        #     d_model=d_model, dropout=dropout, device=self._device
        # )  # TODO: dropout_in? maybe?
        # assert pe_type == 'relative'

        #self.dropout = nn.Dropout(p=dropout_in)

        self.layers = nn.ModuleList(
            [
#                copy.deepcopy(
                    ConformerEncoderBlock( #why deepcopy here?!
                        d_model=d_model,
                        d_ff=d_ff,
                        n_heads=n_heads,
                        kernel_size=kernel_size,
                        dropout=dropout,
                        dropout_att=dropout_att,
                        dropout_layer=dropout_layer,
                        layer_norm_eps=layer_norm_eps,
                        ffn_activation=ffn_activation,
                        param_init=param_init,
                        ffn_bottleneck_dim=ffn_bottleneck_dim,
                        device=self._device,
                    )
#                )
                for _ in range(n_layers)
            ]
        )
        self.norm_out = LayerNorm(d_model, eps=layer_norm_eps)

        self._odim = d_model

        if last_proj_dim > 0 and last_proj_dim != self.output_dim:
            self.bridge = nn.Linear(self._odim, last_proj_dim)
            self._odim = last_proj_dim

        if lstm_hidden_size > 0:
            self.lstm = nn.LSTM(
                input_size=self._odim,
                hidden_size=lstm_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
            self._odim = lstm_hidden_size
        else:
            self.lstm = None

        #self.reset_parameters(param_init)

        # for p in self.parameters():
        #     if p.dim() == 1:
        #         p.data.zero_()

        # for m in self.modules():
        #     if isinstance(m, (torch.nn.Embedding, LayerNorm)):
        #         m.reset_parameters()

        # self.apply(lambda x: init_weights(x, mode=init_mode))
        self.to(self._device)

    # def reset_parameters(self, param_init):
    #     """Initialize parameters."""
    #     if param_init == 'xavier_uniform':
    #         logging.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
    #         if self.conv is None:
    #             nn.init.xavier_uniform_(self.embed.weight)
    #             nn.init.constant_(self.embed.bias, 0.0)
    #         if self.bridge is not None:
    #             nn.init.xavier_uniform_(self.bridge.weight)
    #             nn.init.constant_(self.bridge.bias, 0.0)

    def forward(self, audio_signal, length=None):
        ## type: (Tensor, Optional[Tensor]) -> Tensor, Optional[Tensor]

        audio_signal = torch.transpose(audio_signal, 1, 2)

        # bs, xmax, idim = audio_signal.size()

        if self.conv is None:
            audio_signal = self.embed(audio_signal)
        else:
            # Path through CNN blocks
            # audio_signal1, length1 = self.conv(audio_signal, length)
            audio_signal, length = self.conv(audio_signal, length)

        audio_signal, pos_embs = self.pos_enc(audio_signal)

        # audio_signal = audio_signal2
        # length = length2

        bs, xmax, idim = audio_signal.size()

        # audio_signal = audio_signal * self.scale  # why really? A trick XL-Transformer applied!

        # Create the self-attention mask
        pad_mask = make_pad_mask(length, max_time=xmax, device=self._device)
        xx_mask = pad_mask.unsqueeze(2).repeat([1, 1, xmax])
        #pad_mask = (~pad_mask).unsqueeze(2).repeat(1, 1, idim)his err
        pad_mask = (~pad_mask).unsqueeze(2)

        #pos_idxs = torch.arange(xmax - 1, -1, -1.0, dtype=torch.float)
        #pos_embs = self.pos_emb(pos_idxs)

        #audio_signal = self.dropout(audio_signal)
        #pos_embs = self.pos_emb(pos_idxs)

        for lth, layer in enumerate(self.layers):
            #audio_signal = layer(xs=audio_signal, xx_mask=xx_mask, pos_embs=pos_embs, pad_mask=pad_mask)
            audio_signal = layer(xs=audio_signal, xx_mask=xx_mask, pos_embs=pos_embs, pad_mask=pad_mask)
            # audio_signal.masked_fill_(pad_mask, 0.0)

            # if not self.training:
            #     self.aws_dict['xx_aws_layer%d' % lth] = tensor2np(xx_aws)

        #audio_signal = self.norm_out(audio_signal)

        # Bridge layer
        if self.bridge is not None:
            audio_signal = self.bridge(audio_signal)

        if self.lstm is not None:
            audio_signal, _ = self.lstm(audio_signal)

        audio_signal = audio_signal.masked_fill(pad_mask, 0.0)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        # if length is None:
        #     return audio_signal
        # else:
        return audio_signal, length


def make_pad_mask(seq_lens, max_time, device=None):
    """Make mask for padding.
    Args:
        seq_lens (IntTensor): `[B]`
        device_id (int):
    Returns:
        mask (IntTensor): `[B, T]`
    """
    bs = seq_lens.size(0)
    # max_time = max(seq_lens)

    seq_range = torch.arange(0, max_time, dtype=torch.int32)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_time)
    seq_lens = seq_lens.type(seq_range_expand.dtype).to(seq_range_expand.device)
    # seq_length_expand = seq_range_expand.new(seq_lens).unsqueeze(-1)
    seq_length_expand = seq_lens.unsqueeze(-1)
    mask = seq_range_expand < seq_length_expand

    if device:
        mask = mask.to(device)
    return mask



class PositionalEncoding(torch.nn.Module):
    """Positional encoding.
    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    :param reverse: whether to reverse the input position
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        #self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class RelPositionalEncoding(PositionalEncoding):
    """Relitive positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class.
        :param int d_model: embedding dim
        :param float dropout_rate: dropout rate
        :param int max_len: maximum input length
        """
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self, x):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
        Returns:
            torch.Tensor: x. Its shape is (batch, time, ...)
            torch.Tensor: pos_emb. Its shape is (1, time, ...)
        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[:, : x.size(1)]
        return self.dropout(x), pos_emb #self.dropout(pos_emb)
