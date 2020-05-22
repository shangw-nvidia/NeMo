# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nemo
from .parts import rnn
from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs

logging = nemo.logging


class RNNTEncoder(TrainableNM):
    """A Recurrent Neural Network Transducer (RNN-T).
    Args:
        in_features: Number of input features per step per batch.
        vocab_size: Number of output symbols (inc blank).
        forget_gate_bias: Total initialized value of the bias used in the
            forget gate. Set to None to use PyTorch's default initialisation.
            (See: http://proceedings.mlr.press/v37/jozefowicz15.pdf)
        batch_norm: Use batch normalization in encoder and prediction network
            if true.
        encoder_n_hidden: Internal hidden unit size of the encoder.
        encoder_rnn_layers: Encoder number of layers.
        pred_n_hidden:  Internal hidden unit size of the prediction network.
        pred_rnn_layers: Prediction network number of layers.
        joint_n_hidden: Internal hidden unit size of the joint network.
        rnn_type: string. Type of rnn in SUPPORTED_RNNS.
    """

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

    def __init__(
        self, rnnt: Dict[str, Any], feat_in: int, normalization_mode: Optional[str] = None, frame_splicing: int = 1,
    ):
        super().__init__()
        feat_in = feat_in * frame_splicing

        # Required arguments
        self.encoder_n_hidden = rnnt["encoder_hidden"]
        self.encoder_pre_rnn_layers = rnnt["encoder_pre_rnn_layers"]
        self.encoder_post_rnn_layers = rnnt["encoder_post_rnn_layers"]

        # Optional arguments
        forget_gate_bias = rnnt.get('forget_gate_bias', 1.0)
        encoder_stack_time_factor = rnnt.get('encoder_stack_time_factor', 1)
        dropout = rnnt.get('dropout', 0.0)

        self.encoder = self._encoder(
            feat_in,
            encoder_n_hidden=self.encoder_n_hidden,
            encoder_pre_rnn_layers=self.encoder_pre_rnn_layers,
            encoder_post_rnn_layers=self.encoder_post_rnn_layers,
            forget_gate_bias=forget_gate_bias,
            norm=normalization_mode,
            encoder_stack_time_factor=encoder_stack_time_factor,
            dropout=dropout,
        )
        self.to(self._device)

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # audio_signal: (B, channels, seq_len)
        x = audio_signal.transpose(1, 2).transpose(0, 1)  # (seq_len, B, channels)
        f, x_lens = self.encode(x, length)  # (B, seq_len, channels)
        out = f.transpose(1, 2)  # (B, channels, seq_len)

        return out, x_lens

    def encode(self, x: torch.Tensor, x_lens: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            x: tuple of ``(input, input_lens)``. ``input`` has shape (T, B, I),
                ``input_lens`` has shape ``(B,)``.
        Returns:
            f: tuple of ``(output, output_lens)``. ``output`` has shape
                (B, T, H), ``output_lens``
        """
        x, _ = self.encoder["pre_rnn"](x, None)
        x, x_lens = self.encoder["stack_time"]((x, x_lens))
        x, _ = self.encoder["post_rnn"](x, None)

        return x.transpose(0, 1), x_lens

    def _encoder(
        self,
        in_features,
        encoder_n_hidden,
        encoder_pre_rnn_layers,
        encoder_post_rnn_layers,
        forget_gate_bias,
        norm,
        encoder_stack_time_factor,
        dropout,
    ):
        layers = torch.nn.ModuleDict(
            {
                "pre_rnn": rnn.rnn(
                    input_size=in_features,
                    hidden_size=encoder_n_hidden,
                    num_layers=encoder_pre_rnn_layers,
                    norm=norm,
                    forget_gate_bias=forget_gate_bias,
                    dropout=dropout,
                ),
                "stack_time": rnn.StackTime(factor=encoder_stack_time_factor),
                "post_rnn": rnn.rnn(
                    input_size=encoder_stack_time_factor * encoder_n_hidden,
                    hidden_size=encoder_n_hidden,
                    num_layers=encoder_post_rnn_layers,
                    norm=norm,
                    forget_gate_bias=forget_gate_bias,
                    norm_first_rnn=True,
                    dropout=dropout,
                ),
            }
        )
        return layers


class RNNTDecoder(TrainableNM):
    """A Recurrent Neural Network Transducer (RNN-T).
    Args:
        in_features: Number of input features per step per batch.
        vocab_size: Number of output symbols (inc blank).
        forget_gate_bias: Total initialized value of the bias used in the
            forget gate. Set to None to use PyTorch's default initialisation.
            (See: http://proceedings.mlr.press/v37/jozefowicz15.pdf)
        batch_norm: Use batch normalization in encoder and prediction network
            if true.
        pred_n_hidden:  Internal hidden unit size of the prediction network.
        pred_rnn_layers: Prediction network number of layers.
        rnn_type: string. Type of rnn in SUPPORTED_RNNS.
    """

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
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "target_length": NeuralType(tuple('B'), LengthsType()),
            # "state_hidden": NeuralType(('B', 'D'), RNNStateType()),
            # "state_context": NeuralType(('B', 'D'), RNNStateType())
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
            "outputs": NeuralType(('B', 'D', 'T'), EmbeddedTextType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self, rnnt: Dict[str, Any], num_classes: int, normalization_mode: Optional[str] = None,
    ):
        super().__init__()

        # Required arguments
        self.pred_hidden = rnnt['pred_hidden']
        pred_rnn_layers = rnnt["pred_rnn_layers"]

        # Optional arguments
        forget_gate_bias = rnnt.get('forget_gate_bias', 1.0)
        dropout = rnnt.get('dropout', 0.0)

        self.prediction = self._predict(
            num_classes + 1,  # add 1 for blank symbol
            pred_n_hidden=self.pred_hidden,
            pred_rnn_layers=pred_rnn_layers,
            forget_gate_bias=forget_gate_bias,
            norm=normalization_mode,
            dropout=dropout,
        )
        self.to(self._device)

    def forward(self, targets, target_length):
        # y: (B, U)
        y = rnn.label_collate(targets)

        g, _ = self.predict(y, state=None)  # (B, U + 1, D)
        out = g.transpose(1, 2)  # (B, D, U + 1)

        return out, target_length

    def predict(self, y: Optional[torch.Tensor], state: Optional[torch.Tensor] = None, add_sos: bool = True):
        """
        B - batch size
        U - label length
        H - Hidden dimension size
        L - Number of decoder layers = 2
        Args:
            y: (B, U)
        Returns:
            Tuple (g, hid) where:
                g: (B, U + 1, H)
                hid: (h, c) where h is the final sequence hidden state and c is
                    the final cell state:
                        h (tensor), shape (L, B, H)
                        c (tensor), shape (L, B, H)
        """
        if y is not None:
            # (B, U) -> (B, U, H)
            y = self.prediction["embed"](y)
        else:
            B = 1 if state is None else state[0].size(1)
            y = torch.zeros((B, 1, self.pred_hidden)).to(device=self._device)

        # preprend blank "start of sequence" symbol
        if add_sos:
            B, U, H = y.shape
            start = torch.zeros((B, 1, H)).to(device=y.device, dtype=y.dtype)
            y = torch.cat([start, y], dim=1).contiguous()  # (B, U + 1, H)
        else:
            start = None  # makes del call later easier

        # if state is None:
        #    batch = y.size(0)
        #    state = [
        #        (torch.zeros(batch, self.pred_n_hidden, dtype=y.dtype, device=y.device),
        #         torch.zeros(batch, self.pred_n_hidden, dtype=y.dtype, device=y.device))
        #        for _ in range(self.pred_rnn_layers)
        #    ]

        y = y.transpose(0, 1)  # .contiguous()   # (U + 1, B, H)
        g, hid = self.prediction["dec_rnn"](y, state)
        g = g.transpose(0, 1)  # .contiguous()   # (B, U + 1, H)
        del y, start, state
        return g, hid

    def _predict(self, vocab_size, pred_n_hidden, pred_rnn_layers, forget_gate_bias, norm, dropout):
        layers = torch.nn.ModuleDict(
            {
                "embed": torch.nn.Embedding(vocab_size - 1, pred_n_hidden),
                "dec_rnn": rnn.rnn(
                    input_size=pred_n_hidden,
                    hidden_size=pred_n_hidden,
                    num_layers=pred_rnn_layers,
                    norm=norm,
                    forget_gate_bias=forget_gate_bias,
                    dropout=dropout,
                ),
            }
        )
        return layers


class RNNTJoint(TrainableNM):
    """A Recurrent Neural Network Transducer (RNN-T).
    Args:
        in_features: Number of input features per step per batch.
        vocab_size: Number of output symbols (inc blank).
        forget_gate_bias: Total initialized value of the bias used in the
            forget gate. Set to None to use PyTorch's default initialisation.
            (See: http://proceedings.mlr.press/v37/jozefowicz15.pdf)
        batch_norm: Use batch normalization in encoder and prediction network
            if true.
        encoder_n_hidden: Internal hidden unit size of the encoder.
        encoder_rnn_layers: Encoder number of layers.
        pred_n_hidden:  Internal hidden unit size of the prediction network.
        pred_rnn_layers: Prediction network number of layers.
        joint_n_hidden: Internal hidden unit size of the joint network.
        rnn_type: string. Type of rnn in SUPPORTED_RNNS.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "encoder_outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "decoder_outputs": NeuralType(('B', 'D', 'T'), EmbeddedTextType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            "outputs": NeuralType(('B', 'T', 'D', 'D'), LogitsType()),
        }

    def __init__(
        self,
        rnnt: Dict[str, Any],
        num_classes: int
    ):
        super().__init__()

        # Required arguments
        encoder_hidden = rnnt["encoder_hidden"]
        pred_hidden = rnnt['pred_hidden']
        joint_hidden = rnnt["joint_hidden"]

        # Optional arguments
        dropout = rnnt.get('dropout', 0.0)

        self.pred, self.enc, self.joint_net = self._joint_net(
            vocab_size=num_classes + 1,  # add 1 for blank symbol
            pred_n_hidden=pred_hidden,
            enc_n_hidden=encoder_hidden,
            joint_n_hidden=joint_hidden,
            dropout=dropout,
        )

        self.to(self._device)

    def forward(self, encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor) -> torch.Tensor:
        # encoder = (B, D, T)
        # decoder = (B, D, U + 1)
        encoder_outputs.transpose_(1, 2)  # (B, T, D)
        decoder_outputs.transpose_(1, 2)  # (B, U + 1, D)

        out = self.joint(encoder_outputs, decoder_outputs)  # [B, T, U, K + 1]

        encoder_outputs.transpose_(1, 2)  # (B, D, T)
        decoder_outputs.transpose_(1, 2)  # (B, D, U + 1)
        return out

    def joint(self, f: torch.Tensor, g: torch.Tensor):
        """
        f should be shape (B, T, H)
        g should be shape (B, U + 1, H)
        returns:
            logits of shape (B, T, U, K + 1)
        """
        f = self.enc(f)
        f = f.unsqueeze(dim=2)  # (B, T, 1, D)

        g = self.pred(g)
        g = g.unsqueeze(dim=1)  # (B, 1, U + 1, H)

        # print("f", f.shape, "g", g.shape)

        # inp = torch.cat([f, g], dim=3)  # (B, T, U, H + H2)
        inp = f + g

        del f, g
        # print("inp to jointnet shape :", inp.shape)

        res = self.joint_net(inp)

        # print("joint res", res.shape)

        del inp
        return res

    def _joint_net(self, vocab_size, pred_n_hidden, enc_n_hidden, joint_n_hidden, dropout):
        pred = torch.nn.Linear(pred_n_hidden, joint_n_hidden)
        enc = torch.nn.Linear(enc_n_hidden, joint_n_hidden)

        layers = (
            [torch.nn.Tanh()]
            + ([torch.nn.Dropout(p=dropout)] if dropout else [])
            + [torch.nn.Linear(joint_n_hidden, vocab_size)]
        )
        return pred, enc, torch.nn.Sequential(*layers)
