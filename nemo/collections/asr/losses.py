# Copyright (c) 2019 NVIDIA Corporation
from typing import Tuple

import torch
import torch.nn as nn

from nemo.backends.pytorch.nm import LossNM
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs

try:
    from warprnnt_pytorch import RNNTLoss as WarpRNNTLoss
    WARP_RNNT_LOSS_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    WARP_RNNT_LOSS_AVAILABLE = False


class CTCLossNM(LossNM):
    """
    Neural Module wrapper for pytorch's ctcloss
    Args:
        num_classes (int): Number of characters in ASR model's vocab/labels.
            This count should not include the CTC blank symbol.
        zero_infinity (bool): Whether to zero infinite losses and the associated gradients.
            By default, it is False. Infinite losses mainly occur when the inputs are too
            short to be aligned to the targets.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "log_probs": NeuralType({1: AxisType(TimeTag), 0: AxisType(BatchTag), 2: AxisType(ChannelTag),}),
            # "targets": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "input_length": NeuralType({0: AxisType(BatchTag)}),
            # "target_length": NeuralType({0: AxisType(BatchTag)}),
            "log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "input_length": NeuralType(tuple('B'), LengthsType()),
            "target_length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        loss:
            NeuralType(None)
        """
        # return {"loss": NeuralType(None)}
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, num_classes, zero_infinity=False):
        super().__init__()

        self._blank = num_classes
        self._criterion = nn.CTCLoss(blank=self._blank, reduction='none', zero_infinity=zero_infinity)

    def _loss(self, log_probs, targets, input_length, target_length):
        input_length = input_length.long()
        target_length = target_length.long()
        targets = targets.long()
        loss = self._criterion(log_probs.transpose(1, 0), targets, input_length, target_length)
        # note that this is different from reduction = 'mean'
        # because we are not dividing by target lengths
        loss = torch.mean(loss)
        return loss

    def _loss_function(self, **kwargs):
        return self._loss(*(kwargs.values()))


class RNNTLoss(LossNM):
    """Wrapped :py:class:`warprnnt_pytorch.RNNTLoss`.
    Args:
        blank: Index of the blank label.
        reduction: (string) Specifies the reduction to apply to the output:
            none:
                No reduction will be applied.
            mean:
                The output losses will be divided by the target lengths and
                then the mean over the batch is taken.
            sum:
                Sum all losses in a batch.
    Attributes:
        rnnt_loss: A :py:class:`warprnnt_pytorch.RNNTLoss` instance.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "log_probs": NeuralType({1: AxisType(TimeTag), 0: AxisType(BatchTag), 2: AxisType(ChannelTag),}),
            # "targets": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "input_length": NeuralType({0: AxisType(BatchTag)}),
            # "target_length": NeuralType({0: AxisType(BatchTag)}),
            "log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "input_length": NeuralType(tuple('B'), LengthsType()),
            "target_length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        loss:
            NeuralType(None)
        """
        # return {"loss": NeuralType(None)}
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, num_classes: int, reduction='mean'):
        super().__init__()

        if not WARP_RNNT_LOSS_AVAILABLE:
            raise ValueError("RNNTLoss could not be imported. Please make sure that "
                             "RNNTLoss is properly installed by following the "
                             "instructions at https://github.com/HawkAaron/warp-transducer")

        self._blank = num_classes
        self.rnnt_loss = WarpRNNTLoss(blank=self._blank, reduction=reduction)

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Computes RNNT loss.

        Args:
            inputs: A tuple where the first element is the unnormalized network
                :py:class:`torch.Tensor` outputs of size ``[batch, max_seq_len,
                max_output_seq_len + 1, vocab_size + 1)``. The second element
                is a Tuple of two :py:class:`torch.Tensor`s both of
                size ``[batch]`` that contain the lengths of a) the audio features
                logits and b) the target sequence logits.
            targets: A tuple where the first element is a
                :py:class:`torch.Tensor` such that each entry in the target
                sequence is a class index. Target indices cannot be the blank
                index. It must have size ``[batch, max_seq_len]``. In the former
                form each target sequence is padded to the length of the longest
                sequence and stacked.
                The second element is a :py:class:`torch.Tensor` that gives
                the lengths of the targets. Lengths are specified for each
                sequence to achieve masking under the assumption that sequences
                are padded to equal lengths.
        """

        logits, logit_lens = inputs
        y, y_lens = targets

        # cast to required types
        if logits.dtype != torch.float:
            logits_orig = logits
            logits = logits.float()
            del logits_orig  # save memory *before* computing the loss

        if y.dtype != torch.int32:
            y = y.int()

        if logit_lens.dtype != torch.int32:
            logit_lens = logit_lens.int()

        if y_lens.dtype != torch.int32:
            y_lens = y_lens.int()

        loss = self.rnnt_loss(
            acts=logits, labels=y, act_lens=logit_lens, label_lens=y_lens
        )

        # del new variables that may have been created due to float/int/cuda()
        del logits, y, logit_lens, y_lens, inputs, targets

        return loss
