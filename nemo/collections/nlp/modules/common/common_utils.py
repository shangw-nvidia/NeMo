# Copyright 2020 NVIDIA. All Rights Reserved.
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


from typing import List, Optional

from nemo import logging
from nemo.collections.nlp.modules.common.huggingface.huggingface_utils import (
    get_huggingface_lm_model,
    get_huggingface_lm_models_list,
)

try:
    __megatron_utils_satisfied = True
    from nemo.collections.nlp.modules.common import MegatronBERT, get_megatron_lm_models_list
except Exception as e:
    logging.error('Failed to import Megatron Neural Module and utils: `{}` ({})'.format(str(e), type(e)))
    __megatron_utils_satisfied = False

__all__ = ['get_pretrained_lm_models_list', 'get_pretrained_lm_model']


def get_pretrained_lm_models_list() -> List[str]:
    '''
    Returns the list of support pretrained models
    '''
    if __megatron_utils_satisfied:
        return get_megatron_lm_models_list() + get_huggingface_lm_models_list()
    else:
        return get_huggingface_lm_models_list()


def get_pretrained_lm_model(
    pretrained_model_name: str, config_file: Optional[str] = None, checkpoint_file: Optional[str] = None
):
    '''
    Returns pretrained model
    Args:
        pretrained_model_name (str): pretrained model name, for example, bert-base-uncased.
            See the full list by calling get_pretrained_lm_models_list()
        config_file (str): path to the model configuration file
        checkpoint_file (str): path to the pretrained model checkpoint
    Returns:
        Pretrained model (NM)
    '''
    if pretrained_model_name in get_huggingface_lm_models_list():
        model = get_huggingface_lm_model(config_file=config_file, pretrained_model_name=pretrained_model_name)
    # elif __megatron_utils_satisfied and pretrained_model_name in get_megatron_lm_models_list():
    #     if pretrained_model_name == 'megatron-bert-cased' or pretrained_model_name == 'megatron-bert-uncased':
    #         if not (config and checkpoint_file):
    #             raise ValueError(f'Config file and pretrained checkpoint_file required for {pretrained_model_name}')
    #     if not config:
    #         config = get_megatron_config_file(pretrained_model_name)
    #     if isinstance(config, str):
    #         with open(config) as f:
    #             config = json.load(f)
    #     if not vocab:
    #         vocab = get_megatron_vocab_file(pretrained_model_name)
    #     if not checkpoint_file:
    #         checkpoint_file = get_megatron_checkpoint_file(pretrained_model_name)
    #     model = MegatronBERT(
    #         model_name=pretrained_model_name,
    #         vocab_file=vocab,
    #         hidden_size=config['hidden-size'],
    #         num_attention_heads=config['num-attention-heads'],
    #         num_layers=config['num-layers'],
    #         max_seq_length=config['max-seq-length'],
    #     )
    else:
        raise ValueError(f'{pretrained_model_name} is not supported')

    if checkpoint_file:
        model.restore_from(checkpoint_file)
        logging.info(f"{pretrained_model_name} model restored from {checkpoint_file}")
    return model
