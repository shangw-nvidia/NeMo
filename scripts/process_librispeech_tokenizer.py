# Copyright (c) 2019 NVIDIA Corporation
#
# USAGE: python get_librispeech_data.py --data_root=<where to put data>
#        --data_set=<datasets_to_download>
# where <datasets_to_download> can be: dev_clean, dev_other, test_clean,
# test_other, train_clean_100, train_clean_360, train_other_500 or ALL
# You can also put more than one data_set comma-separated:
# --data_set=dev_clean,train_clean_100
import argparse
import json
import logging
import os

import tokenizers
from tqdm import tqdm

"""
python process_librispeech_tokenizer.py --manifest="/home/smajumdar/PycharmProjects/nemo-eval/nemo_eval/librispeech/manifests/train_clean_100.json,/home/smajumdar/PycharmProjects/nemo-eval/nemo_eval/librispeech/manifests/train_clean_360.json,/home/smajumdar/PycharmProjects/nemo-eval/nemo_eval/librispeech/manifests/train_other_500.json" --data_root="/home/smajumdar/PycharmProjects/nemo-eval/nemo_eval/librispeech/manifests/" --log
"""

parser = argparse.ArgumentParser(description='LibriSpeech Data download')
parser.add_argument("--manifest", required=True, default=None, type=str)
parser.add_argument("--data_root", required=True, default=None, type=str)
parser.add_argument("--vocab_size", default=1024, type=int)
parser.add_argument("--tokenizer", default="bpe", choices=["bpe", "wpe"])
parser.add_argument("--log", action='store_true')
parser.set_defaults(log=False)
args = parser.parse_args()


def __build_document_from_manifests(data_root: str, manifests: str,):
    if ',' in manifests:
        manifests = manifests.split(',')
    else:
        manifests = [manifests]

    document_dir = os.path.join(data_root, 'librispeech_doc')
    if not os.path.exists(document_dir):
        os.makedirs(document_dir)

    document_path = os.path.join(document_dir, 'document.txt')

    if os.path.exists(document_path):
        logging.info('Corpus already exists at path : %s', document_path)
        return document_path

    with open(document_path, 'w') as out_writer:
        for manifest in manifests:
            with open(manifest, 'r') as in_reader:
                for line in in_reader:
                    item = json.loads(line)
                    text = item['text']

                    out_writer.write(text + '\n')
                    out_writer.flush()

            logging.info(f"Finished extracting manifest : {manifest}")

        logging.info("Finished extracting all manifests !")
    return document_path


def __process_data(text_path: str, dst_folder: str, vocab_size: int, tokenizer_type: str):
    """
    Converts flac to wav and build manifests's json
    Args:
        text_path: source with text lines
        dst_folder: where wav files will be stored
        vocab_size: vocabular size used in encoding the text
        tokenizer_type: type of tokenization to perform - bpe or wpe

    Returns:

    """
    tokenizer_dir = os.path.join(dst_folder, 'librispeech_tokenizer_{}_v{}').format(tokenizer_type, vocab_size)

    if not os.path.exists(tokenizer_dir):
        os.makedirs(tokenizer_dir)

    if tokenizer_type == 'bpe':
        tokenizer = tokenizers.ByteLevelBPETokenizer(lowercase=True)
    else:
        tokenizer = tokenizers.BertWordPieceTokenizer(lowercase=True)

    tokenizer.train(text_path, vocab_size=vocab_size)

    tokenizer.save(tokenizer_dir)

    return tokenizer_dir


def main():
    data_root = args.data_root
    manifests = args.manifest
    vocab_size = args.vocab_size
    tokenizer = args.tokenizer

    data_root = os.path.join(data_root, 'LibriSpeech')

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    if args.log:
        logging.basicConfig(level=logging.INFO)

    text_corpus_path = __build_document_from_manifests(data_root, manifests)
    tokenizer_path = __process_data(text_corpus_path, data_root, vocab_size, tokenizer)

    print("Serialized tokenizer at location :", tokenizer_path)
    logging.info('Done!')


if __name__ == "__main__":
    main()
