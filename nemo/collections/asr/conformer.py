# Copyright (c) 2019 NVIDIA Corporation
import copy
import math
from typing import Optional

import torch
import torch.nn as nn

from .parts.conformer import ConvEncoder, ConformerEncoderBlock, Conv2dSubsampling
from .parts.conformer import XLPositionalEmbedding
from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import *
from nemo.utils import logging
from nemo.utils.decorators import add_port_docs


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

    def _prepare_for_deployment(self):
        m_count = 0
        for m in self.modules():
            if type(m).__name__ == "MaskedConv1d":
                m.use_mask = False
                m_count += 1
        logging.warning(f"Turned off {m_count} masked convolutions")

        input_example = torch.randn(16, self.__feat_in, 256)
        return input_example, None

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
        n_layers_sub1,
        n_layers_sub2,
        n_layers,
        d_model,
        n_heads,
        pe_type,
        chunk_size_left,
        chunk_size_current,
        chunk_size_right,
        task_specific_layer,
        conv_channels,
        n_splices,
        n_stacks,
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
        #feat_out,
    ):
        super().__init__()

        if n_layers_sub1 < 0 or (n_layers_sub1 > 1 and n_layers < n_layers_sub1):
            raise ValueError('Set n_layers_sub1 between 1 to n_layers.')
        if n_layers_sub2 < 0 or (n_layers_sub2 > 1 and n_layers_sub1 < n_layers_sub2):
            raise ValueError('Set n_layers_sub2 between 1 to n_layers_sub1.')

        d_ff = d_model * ff_expansion_factor

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pe_type = pe_type

        # for streaming TransformerXL encoder
        #self.chunk_size_left = chunk_size_left
        #self.chunk_size_current = chunk_size_current
        #self.chunk_size_right = chunk_size_right
        #self.latency_controlled = chunk_size_left > 0 or chunk_size_current > 0 or chunk_size_right > 0
        #self.scale = math.sqrt(d_model)

        # for hierarchical encoder
        #self.n_layers_sub1 = n_layers_sub1
        #self.n_layers_sub2 = n_layers_sub2
        #self.task_specific_layer = task_specific_layer

        # for bridge layers
        self.bridge = None
        #self.bridge_sub1 = None
        #self.bridge_sub2 = None

        # for attention plot
        #self.aws_dict = {}
        #self.data_dict = {}

        # Setting for CNNs
        if conv_channels:
            assert n_stacks == 1 and n_splices == 1

            # self.conv = Conv2dSubsampling(
            #     idim=input_dim,
            #     odim=d_model,
            #     conv_channels=32,
            #     kernel_size=3,
            #     dropout_rate=0,
            #     activation=nn.ReLU(),
            #     rel_pos=False,
            # )
            # self._odim = d_model

            self.conv = ConvEncoder(
                input_dim,
                in_channel=conv_in_channel,
                channels=conv_channels,
                kernel_sizes=conv_kernel_sizes,
                strides=conv_strides,
                poolings=conv_poolings,
                dropout=0.0,
                batch_norm=conv_batch_norm,
                layer_norm=conv_layer_norm,
                layer_norm_eps=layer_norm_eps,
                residual=False,
                bottleneck_dim=d_model,
                param_init=conv_param_init,
                device=self._device,
            )
            self._odim = self.conv.output_dim
        else:
            self.conv = None
            self._odim = input_dim * n_splices * n_stacks
            self.embed = nn.Linear(self._odim, d_model)

        # calculate subsampling factor
        # self._factor = 1
        # if self.conv is not None:
        #     self._factor *= self.conv.subsampling_factor

        self.pos_emb = XLPositionalEmbedding(d_model=d_model, dropout=dropout, device=self._device)  # TODO: dropout_in? maybe?
        assert pe_type == 'relative'

        self.dropout = nn.Dropout(p=dropout_in)

        self.layers = nn.ModuleList(
            [
                copy.deepcopy(
                    ConformerEncoderBlock(
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
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self._odim = d_model

        # if n_layers_sub1 > 0:
        #     if task_specific_layer:
        #         self.layer_sub1 = ConformerEncoderBlock(
        #             d_model,
        #             d_ff,
        #             n_heads,
        #             kernel_size,
        #             dropout,
        #             dropout_att,
        #             dropout_layer,
        #             layer_norm_eps,
        #             ffn_activation,
        #             param_init,
        #             ffn_bottleneck_dim=ffn_bottleneck_dim,
        #         )
        #     self.norm_out_sub1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        #     if last_proj_dim > 0 and last_proj_dim != self.output_dim:
        #         self.bridge_sub1 = nn.Linear(self._odim, last_proj_dim)
        #
        # if n_layers_sub2 > 0:
        #     if task_specific_layer:
        #         self.layer_sub2 = ConformerEncoderBlock(
        #             d_model,
        #             d_ff,
        #             n_heads,
        #             kernel_size,
        #             dropout,
        #             dropout_att,
        #             dropout_layer,
        #             layer_norm_eps,
        #             ffn_activation,
        #             param_init,
        #             ffn_bottleneck_dim=ffn_bottleneck_dim,
        #         )
        #     self.norm_out_sub2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        #     if last_proj_dim > 0 and last_proj_dim != self.output_dim:
        #         self.bridge_sub2 = nn.Linear(self._odim, last_proj_dim)

        if last_proj_dim > 0 and last_proj_dim != self.output_dim:
            self.bridge = nn.Linear(self._odim, last_proj_dim)
            self._odim = last_proj_dim

        self.reset_parameters(param_init)

        # self.apply(lambda x: init_weights(x, mode=init_mode))
        self.to(self._device)

    def reset_parameters(self, param_init):
        """Initialize parameters."""
        if param_init == 'xavier_uniform':
            logging.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
            if self.conv is None:
                nn.init.xavier_uniform_(self.embed.weight)
                nn.init.constant_(self.embed.bias, 0.0)
            if self.bridge is not None:
                nn.init.xavier_uniform_(self.bridge.weight)
                nn.init.constant_(self.bridge.bias, 0.0)
            # if self.bridge_sub1 is not None:
            #     nn.init.xavier_uniform_(self.bridge_sub1.weight)
            #     nn.init.constant_(self.bridge_sub1.bias, 0.0)
            # if self.bridge_sub2 is not None:
            #     nn.init.xavier_uniform_(self.bridge_sub2.weight)
            #     nn.init.constant_(self.bridge_sub2.bias, 0.0)

    def forward(self, audio_signal, length=None):
        ## type: (Tensor, Optional[Tensor]) -> Tensor, Optional[Tensor]

        # s_input, length = self.encoder(([audio_signal], length))

        # eouts = {'ys': {'xs': None, 'xlens': None},
        #          'ys_sub1': {'xs': None, 'xlens': None},
        #          'ys_sub2': {'xs': None, 'xlens': None}}

        audio_signal = torch.transpose(audio_signal, 1, 2)
        # N_l = self.chunk_size_left
        # N_c = self.chunk_size_current
        # N_r = self.chunk_size_right
        #
        # bs, xmax, idim = audio_signal.size()

        # if self.latency_controlled:
        #     audio_signal = blockwise(audio_signal, N_l, N_c, N_r)

        if self.conv is None:
            audio_signal = self.embed(audio_signal)
        else:
            # Path through CNN blocks
            audio_signal, length = self.conv(audio_signal, length)

        # if not self.training:
        #     self.data_dict['elens'] = tensor2np(length)

        # if self.latency_controlled:
        #     # streaming Conformer encoder
        #     _N_l = max(0, N_l // self.subsampling_factor)
        #     _N_c = N_c // self.subsampling_factor
        #
        #     n_blocks = audio_signal.size(0) // bs
        #
        #     emax = xmax // self.subsampling_factor
        #     if xmax % self.subsampling_factor != 0:
        #         emax += 1
        #
        #     audio_signal = audio_signal * self.scale
        #     pos_idxs = torch.arange(audio_signal.size(1) - 1, -1, -1.0, dtype=torch.float)
        #     pos_embs = self.pos_emb(pos_idxs, self._device)
        #
        #     xx_mask = None  # NOTE: no mask
        #     for lth, layer in enumerate(self.layers):
        #         audio_signal, xx_aws = layer(audio_signal, xx_mask, pos_embs=pos_embs)
        #         if not self.training:
        #             n_heads = xx_aws.size(1)
        #             xx_aws = xx_aws[:, :, _N_l : _N_l + _N_c, _N_l : _N_l + _N_c]
        #             xx_aws = xx_aws.view(bs, n_blocks, n_heads, _N_c, _N_c)
        #             xx_aws_center = xx_aws.new_zeros(bs, n_heads, emax, emax)
        #             for blc_id in range(n_blocks):
        #                 offset = blc_id * _N_c
        #                 emax_blc = xx_aws_center[:, :, offset : offset + _N_c].size(2)
        #                 xx_aws_chunk = xx_aws[:, blc_id, :, :emax_blc, :emax_blc]
        #                 xx_aws_center[:, :, offset : offset + _N_c, offset : offset + _N_c] = xx_aws_chunk
        #             self.aws_dict['xx_aws_layer%d' % lth] = tensor2np(xx_aws_center)
        #
        #     # Extract the center region
        #     audio_signal = audio_signal[:, _N_l : _N_l + _N_c]  # `[B * n_blocks, _N_c, d_model]`
        #     audio_signal = audio_signal.contiguous().view(bs, -1, audio_signal.size(2))
        #     audio_signal = audio_signal[:, :emax]
        #
        # else:
        bs, xmax, idim = audio_signal.size()
        #audio_signal = audio_signal * self.scale  # why really?

        # Create the self-attention mask
        pad_mask = make_pad_mask(length, max_time=xmax, device=self._device)
        xx_mask = pad_mask.unsqueeze(2).repeat([1, 1, xmax])
        pad_mask = (~pad_mask).unsqueeze(2).repeat(1, 1, idim)

        pos_idxs = torch.arange(xmax - 1, -1, -1.0, dtype=torch.float)
        pos_embs = self.pos_emb(pos_idxs)

        audio_signal = self.dropout(audio_signal)

        for lth, layer in enumerate(self.layers):
            audio_signal = layer(xs=audio_signal, xx_mask=xx_mask, pos_embs=pos_embs, pad_mask=pad_mask)
            #audio_signal.masked_fill_(pad_mask, 0.0)

            # if not self.training:
            #     self.aws_dict['xx_aws_layer%d' % lth] = tensor2np(xx_aws)

            # Pick up outputs in the sub task before the projection layer
            # if lth == self.n_layers_sub1 - 1:
            #     xs_sub1 = (
            #         self.layer_sub1(audio_signal, xx_mask, pos_embs=pos_embs)
            #         if self.task_specific_layer
            #         else audio_signal.clone()
            #     )
            #     xs_sub1 = self.norm_out_sub1(xs_sub1)
            #     if self.bridge_sub1 is not None:
            #         xs_sub1 = self.bridge_sub1(xs_sub1)
            #     # if task == 'ys_sub1':
            #     #     eouts[task]['xs'], eouts[task]['xlens'] = xs_sub1, xlens
            #     #     return eouts
            # if lth == self.n_layers_sub2 - 1:
            #     xs_sub2 = (
            #         self.layer_sub2(audio_signal, xx_mask, pos_embs=pos_embs)
            #         if self.task_specific_layer
            #         else audio_signal.clone()
            #     )
            #     xs_sub2 = self.norm_out_sub2(xs_sub2)
            #     if self.bridge_sub2 is not None:
            #         xs_sub2 = self.bridge_sub2(xs_sub2)
            #     # if task == 'ys_sub2':
            #     #     eouts[task]['xs'], eouts[task]['xlens'] = xs_sub2, xlens
            #     #     return eouts

        #audio_signal = self.norm_out(audio_signal)

        # Bridge layer
        if self.bridge is not None:
            audio_signal = self.bridge(audio_signal)

        # if task in ['all', 'ys']:
        #     eouts['ys']['xs'], eouts['ys']['xlens'] = xs, xlens
        # if self.n_layers_sub1 >= 1 and task == 'all':
        #     eouts['ys_sub1']['xs'], eouts['ys_sub1']['xlens'] = xs_sub1, xlens
        # if self.n_layers_sub2 >= 1 and task == 'all':
        #     eouts['ys_sub2']['xs'], eouts['ys_sub2']['xlens'] = xs_sub2, xlens
        # return eouts
        #audio_signal.masked_fill_(pad_mask, 0.0)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        # if length is None:
        #     return audio_signal
        # else:
        return audio_signal, length

        # return s_input[-1], length


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
