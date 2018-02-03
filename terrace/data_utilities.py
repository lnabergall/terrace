""""""

import os
import re
import random

import numpy as np
import spacy
from segtok import segmenter


LANGUAGES = {
    "english": "en",
    "german": "de",
    "spanish": "es",
    "portuguese": "pt",
    "french": "fr",
    "italian": "it",
    "dutch": "nl",
}

PAR_SPLIT_REGEX = re.compile(r"\s{2,}")


def apply_to_collection(function, collection):
    # Applies function on every non-container element of collection.
    if (isinstance(collection, str) or isinstance(collection, bytes) 
            or isinstance(collection, int) or isinstance(collection, float)):
        return function(collection)
    elif isinstance(collection, list):
        return [apply_to_collection(function, element) for element in collection]
    elif isinstance(collection, tuple):
        return tuple(apply_to_collection(function, element) 
                     for element in collection)
    elif isinstance(collection, dict):
        return {key: apply_to_collection(function, value) 
                for key, value in collection.items()}
    elif isinstance(collection, set):
        return set(apply_to_collection(function, list(collection)))
    else:
        return function(collection)


def aggregate_on_collection(function, collection):
    # Recursively applies function on collection and every element of collection.
    if (isinstance(collection, str) or isinstance(collection, bytes) 
            or isinstance(collection, int) or isinstance(collection, float)):
        return function(collection)
    elif isinstance(collection, list):
        return function([aggregate_on_collection(function, element) 
                         for element in collection])
    elif isinstance(collection, tuple):
        return function(tuple(aggregate_on_collection(function, element) 
                              for element in collection))
    elif isinstance(collection, dict):
        return function([aggregate_on_collection(function, value) 
                         for value in collection.values()])
    elif isinstance(collection, set):
        return function(set(aggregate_on_collection(function, list(collection))))
    else:
        return function(collection)


def sample_without_replace(sequence, size):
    sample_indices = random.sample(range(len(sequence)), size)
    sample = [sequence[i] for i in sample_indices]
    sample_indices = set(sample_indices)
    reduced_sequence = [element for i, element in enumerate(sequence) 
                        if i not in sample_indices]
    return sample, reduced_sequence


def partition(sequence, indices):
    if indices[0] != 0:
        indices = [0] + indices
    if indices[-1] != len(sequence):
        indices.append(len(sequence))
    index_pairs = [(indices[i], indices[i+1]) for i in range(len(indices)-1)]
    return [sequence[index1:index2] for index1, index2 in index_pairs]


def count_index(iterable, counter, count):
    total_count = 0
    for i, element in enumerate(iterable):
        total_count += aggregate_on_collection(counter, element)
        if total_count > count:
            break
    if iterable:
        return i
    else:
        return None


def filter_dict(dictionary, keys):
    return {key: value for key, value in dictionary.items() if key in keys}


def save(data, file_path, append=False):
    if append:
        mode = "a"
    else:
        mode = "w"
    with open(file_path, mode) as data_file:
        data_file.write(str(data))


def find_filenames(root_dir, file_pattern, walk=True):
    if isinstance(file_pattern, str):
        regex = re.compile(file_pattern)
    names = []
    if walk:
        for dir_path, dir_names, file_names in os.walk(root_dir):
            names.extend(
                [os.path.join(dir_path, file_name) for file_name in file_names 
                 if regex.fullmatch(os.path.join(dir_path, file_name))])
    else:
        names = [os.path.join(root_dir, file_name) 
                      for file_name in os.listdir(root_dir)
                      if regex.fullmatch(os.path.join(dir_path, file_name))]
    return names


def delete_files(file_paths):
    for file_path in file_paths:
        os.remove(file_path)


def formatted_time(self):
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def get_statistics(sequence):
    return {
        "mean": np.mean(sequence),
        "std": np.std(sequence),
        "median": np.percentile(sequence, 50),
        "upper_quartile": np.percentile(sequence, 75),
        "lower_quartile": np.percentile(sequence, 25), 
        "maximum": np.max(sequence),
        "minimum": np.min(sequence),
    }


def split_string(string, split_into=None, split_on=None, language=None):
    if split_into == "sentences":
        from spacy.util import get_lang_class
        if language is None:
            splitter = segmenter.split_multi
            segments = [sentence for sentence in splitter(string)]
        elif language.lowercase() in LANGUAGES:
            splitter = get_lang_class(LANGUAGES[language.lowercase()])
            segments = [sentence.text for sentence in document.sents]
        else:
            splitter = get_lang_class("xx")  # multi-language model
            segments = [sentence.text for sentence in document.sents]
    elif split_into == "lines":
        segments = string.splitlines()
    elif split_into == "paragraphs":
        segments = PAR_SPLIT_REGEX.split(string)
    elif split_on is not None:
        segments = re.split("|".join(map(re.escape, split_on)), string)
    else:
        raise ValueError("Requires 'split_into' or 'split_on' argument.")

    return segments