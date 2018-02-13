""""""

import os
import sys
import re
import tarfile
import functools

import requests
import torch

from terrace.data import TextDataset, TensorDataset, DataSource


CORPUS_URL = ("http://www.statmt.org/lm-benchmark/"
              "1-billion-word-language-modeling-benchmark-r13output.tar.gz")
ORIG_VOCAB_URL = ("http://download.tensorflow.org/models/LM_LSTM_CNN/"
                  "vocab-2016-09-10.txt")
OOV_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
RESERVED_TOKENS = [PAD_TOKEN, EOS_TOKEN]
PAD_ID = RESERVED_TOKENS.index(PAD_TOKEN)
EOS_ID = RESERVED_TOKENS.index(EOS_TOKEN)


def download_corpus(directory):
    corpus_file_name = os.path.basename(CORPUS_URL)
    corpus_file_path = os.path.join(directory, corpus_file_name)
    if (not os.path.exists(corpus_file_path) 
            and not os.path.exists(corpus_file_path[:-7])):
        # download
        with open(corpus_file_path, "wb") as corpus_file:
            response = requests.get(CORPUS_URL)
            corpus_file.write(response.content)
        # extract
        with tarfile.open(corpus_file_path, "r:gz") as corpus_tar:
            corpus_tar.extractall(directory)


def get_original_vocab(directory):
    vocab_file_name = os.path.basename(ORIG_VOCAB_URL)
    vocab_file_path = os.path.join(directory, vocab_file_name)
    if not os.path.exists(vocab_file_path):
        # download
        with open(vocab_file_path, "wb") as vocab_file:
            response = requests.get(ORIG_VOCAB_URL)
            vocab_file.write(response.content)
    return set([line.strip() for line in open(vocab_file_path, encoding="utf-8")])


def replace_oov(original_vocab, string):
    return " ".join([word if word in original_vocab else OOV_TOKEN 
                     for word in string.split()])


def train_data_filenames(directory):
    return [os.path.join(directory, 
                         "1-billion-word-language-modeling-benchmark-r13output",
                         "training-monolingual.tokenized.shuffled",
                         "news.en-%05d-of-00100" % i) for i in range(1, 100)]


def valid_data_filename(directory):
    return os.path.join(directory, 
                        "1-billion-word-language-modeling-benchmark-r13output",
                        "heldout-monolingual.tokenized.shuffled",
                        "news.en.heldout-00000-of-00050")


def data_reader(file_path):
    return [([], [line.strip()]) for line in open(file_path, encoding="utf-8")]


class SequenceDataset(TextDataset):

    def convert_to_tensor(self, id_count):
        # Assumes the dataset has been converted to IDs
        tensor_data = []
        for i, (input_data, target_data) in enumerate(self.data):
            data_point = []
            for element in [input_data, target_data]:
                if not element:
                    data_point.append(None)
                else:
                    data_point.append(torch.LongTensor(element))
            tensor_data.append(tuple(data_point))

        return tensor_data


def get_1billion_dataset(root_dir, dataset_type="training", 
                         chars=None, vocab_encoder=None):
    if dataset_type.lower().startswith("valid"):
        file_pattern = re.escape(valid_data_filename(root_dir))
    else:
        file_pattern = "|".join(
            [re.escape(file_name) for file_name in train_data_filenames(root_dir)])
    dataset = SequenceDataset.from_storage(root_dir, file_pattern, data_reader, 
                                           name="1billion_word_" + dataset_type)
    if chars is not None:
        dataset.truncate(tokens=chars, token_type="character")

    original_vocab = get_original_vocab(root_dir)
    oov_replacer = functools.partial(replace_oov, original_vocab)
    dataset.clean(cleaner=oov_replacer)

    dataset.convert_to_ids(converter=vocab_encoder)
    if dataset_type == "training":
        # Remove outlier sentences that consume too much memory
        dataset.filter(lambda data_point: len(data_point[1]) < 400)
    # add EOS token
    dataset.map(lambda data_point: (data_point[0], data_point[1] + [EOS_ID]))

    return dataset


def convert_to_datasource(sequence_dataset, vocab_size):
    tensor_data = sequence_dataset.convert_to_tensor(
        len(RESERVED_TOKENS)+vocab_size)
    data_source = DataSource(data=tensor_data, 
                             name=sequence_dataset.name + "_datasource")
    data_source.shuffle()
    return data_source