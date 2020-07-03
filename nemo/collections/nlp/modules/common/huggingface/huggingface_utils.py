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

from transformers import (
    ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    AlbertConfig,
    BertConfig,
    RobertaConfig,
)

from nemo.collections.nlp.modules.common.huggingface.albert import AlbertEncoder
from nemo.collections.nlp.modules.common.huggingface.bert import BertEncoder
from nemo.collections.nlp.modules.common.huggingface.roberta import RobertaEncoder

__all__ = ['MODELS', 'get_huggingface_lm_model', 'get_huggingface_lm_models_list']


def get_huggingface_lm_model(pretrained_model_name: str, config_file: Optional[str] = None):
    '''
    Returns the dict of special tokens associated with the model.
    Args:
        pretrained_mode_name ('str'): name of the pretrained model from the hugging face list,
            for example: bert-base-cased
        config_file: path to model configuration file.
    '''
    model_type = pretrained_model_name.split('-')[0]
    if model_type in MODELS:
        model_class = MODELS[model_type]['class']
        if config_file:
            config_class = MODELS[model_type]['config']
            return model_class(config_class.from_json_file(config_file))
        else:
            return model_class.from_pretrained(pretrained_model_name)
    else:
        raise ValueError(f'{pretrained_model_name} is not supported')


MODELS = {
    'bert': {
        'default': 'bert-base-uncased',
        'class': BertEncoder,
        'config': BertConfig,
        'model_list': BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    },
    'roberta': {
        'default': 'roberta-base',
        'class': RobertaEncoder,
        'config': RobertaConfig,
        'model_list': ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    },
    'albert': {
        'default': 'albert-base-v2',
        'class': AlbertEncoder,
        'config': AlbertConfig,
        'model_list': ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    },
}


def get_huggingface_lm_models_list() -> List[str]:
    '''
    Returns the list of supported HuggingFace models
    '''
    huggingface_models = []
    for model in MODELS:
        model_names = MODELS[model]['model_list']
        huggingface_models.extend(model_names)
    return huggingface_models
