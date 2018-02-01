""""""

import os
import sys
import re
import warnings
import random
from itertools import chain
from collections import Counter, deque

import numpy as np
import torch
from torch.utils.data import TensorDataset as TorchTensorDataset

from . import data_utilities as utils
from . import tensor_utilities as tensor_utils


class Dataset:

    def __init__(self, data, input_features=None, target_features=None, name=None):
        """
        Args:
            data: List; a sequence of 2-tuples representing an input and a target, 
                each of which is a list.
            input_features: Dict; a mapping from indices (of the input part 
                of a data point) to feature names.
            target_features: Dict; a mapping from indices (of the target part 
                of a data point) to feature names.
            name: Str; used for storage and loading (optional, default: None).
        """
        self.data = data
        self.input_features = input_features
        self.target_features = target_features
        self.name = name

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        return self.data.__iter__()

    def __next__(self):
        return self.data.__next__()

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        return Dataset.concatenate(self, other)

    def __repr__(self):
        if self.name is None:
            string = "<Dataset(size={})>".format(len(self))
        else:
            string = "<Dataset(name={}, size={})>".format(self.name, len(self))
        return string

    @classmethod
    def from_storage(cls, root_dir, file_pattern, data_reader, name=None):
        """
        Args:
            root_dir: Str; root directory containing the dataset.
            file_pattern: Str; a regular expression matching 
                the paths of files in the dataset.
            data_reader: Callable; accepts the path to a file and returns 
                a list of data points containing the file's decoded data.
            name: Str; name of the dataset (optional, default: None). 
        """
        file_names = utils.find_filenames(root_dir, file_pattern)
        data = chain.from_iterable(data_reader(file_name) for file_name in file_names)
        return Dataset(data, name=name)

    @classmethod
    def concatenate(cls, *datasets, method="dataset", name=None):
        """
        Args:
            *datasets: Sequence of Datasets. 
            method: Str; concatenation method, accepts 'dataset', where 
                the new dataset contains the union of the sequence of data points 
                from each dataset, or 'point', where the ith data point
                from each dataset is concatenated---this requires that 
                every dataset has the same size (default: 'dataset').
            name: Str; name of the new dataset.
        """
        if method == "dataset":
            data = list(chain.from_iterable([dataset.data for dataset in datasets]))
            input_features = datasets[0].input_features
            target_features = datasets[0].target_features
        elif method == "point":
            if len(set([len(dataset) for dataset in datasets])) != 1:
                raise ValueError(
                    "Method 'point' requires all datasets to be the same size.")
            data = []
            for data_points in zip(*datasets):
                data.append(tuple([element[0] for element in part] 
                                  for part in zip(*data_points)))
            input_features = None
            target_features = None
        else:
            raise ValueError("Provided argument is not a valid method: '" 
                             + method + "'.")

        return Dataset(data, input_features, target_features, name)

    @staticmethod
    def apply_to_point(function, data_point, multi=False, split=False):
        if split:
            # assumes multi is True and either input or target is empty
            new_data_points = []
            for i, element in enumerate(data_point[0] + data_point[1]):
                is_input = True if i < len(data_point[0]) else False
                new_elements = utils.apply_to_collection(function, element)
                new_data_points.extend(
                        [([new_element], []) if is_input else ([], [new_element]) 
                         for new_element in new_elements])
            return new_data_points
        else:
            new_data_point = [[], []]
            for i, element in enumerate(data_point[0] + data_point[1]):
                j = 0 if i < len(data_point[0]) else 1
                if multi:
                    new_data_point[j].extend(
                        utils.apply_to_collection(function, element))
                else:
                    new_data_point[j].append(
                        utils.apply_to_collection(function, element))
            return (new_data_point[0], new_data_point[1])

    @staticmethod
    def datamethod(multi=False, split=False):

        def outer_wrapper(function):

            def inner_wrapper(data_point):
                return Dataset.apply_to_point(function, data_point, multi, split)

            return inner_wrapper

        return outer_wrapper

    def store(self, dir_path, data_writer, file_name_prefix=None, file_count=100):
        """
        Args:
            dir_path: Str; path of a directory where the dataset will be stored.
            data_writer: Callable; accepts a list of data points and a file name 
                and writes the data to file. 
            file_name_prefix: Str; prefix of all file names 
                (optional, default: None). 
            file_count: Int; number of files to use for storage 
                (optional, default: 100).
        """
        if file_name_prefix is None:
            file_name_prefix = self.name
        for i in range(file_count):
            data_chunk = self.data[i*len(self):(i+1)*len(self)]
            file_id = (len(str(file_count)) - len(str(i+1)))*"0" + str(i+1)
            file_name = file_name_prefix + "_" + file_id + "-" + str(file_count)
            file_name = os.path.join(dir_path, file_name)
            data_writer(data_chunk, file_name)

    def sample(self, size, in_place=True, name=None):
        if in_place:
            self.data = random.sample(self.data, k=size)
        else:
            return Dataset(random.sample(self.data, k=size), self.input_features, 
                           self.target_features, name=name)

    def filter(self, predicate, in_place=True, name=None):
        """
        Args:
            predicate: Callable; accepts a data point and returns a boolean.
            in_place: Bool.
            name: Str.
        """
        if in_place:
            self.data = list(filter(predicate, self.data))
        else:
            return Dataset(list(filter(predicate, self.data)), self.input_features, 
                           self.target_features, name=name)

    def truncate(self, size, in_place=True):
        if in_place:
            self.data = self.data[:size]
        else:
            return Dataset(self.data[:size], self.input_features, 
                           self.target_features, name)

    def map(self, mapping, multi=False, memory_efficient=False, 
            in_place=True, name=None):
        """
        Args:
            mapping: Callable; accepts a data point and returns either 
                a new data point (if multi is False) or a list of data points 
                (if multi is True).
            multi: Bool; indicates whether mapping returns a data point 
                or list of data points.
            memory_efficient: Bool; indicates whether to minimize 
                memory consumption or not (not applicable if in_place is False).
            in_place: Bool.
            name: Str.
        """
        if memory_efficient and in_place and not multi:
            for i in range(len(self.data)):
                self.data[i] = mapping(self.data[i])
        else:
            if multi:
                data = list(chain.from_iterable(map(mapping, self.data)))
            else:
                data = list(map(mapping, self.data))
            if in_place:
                self.data = data
            else:
                return Dataset(data, self.input_features, 
                               self.target_features, name)

    def shuffle(self, in_place=True):
        if in_place:
            random.shuffle(self.data)
        else:
            return self.sample(size=len(self), name=self.name)

    def partition(self, partition_spec, names=None):
        """
        Args:
            partition_spec: List; a sequence of floats between 0 and 1 or 
                a sequence of integers, specifying either the percentages 
                of the dataset to place in each part or the exact number
                of data points to place in each part.
            names: List; names of the new datasets (optional, default: None).
        """
        if all(0 <= value <= 1 for value in partition_spec):
            last_index = 0
            indices = []
            for value in partition_spec:
                last_index += int(len(partition_spec)*value)
                indices.append(last_index) 
        else:
            indices = list(np.cumsum(partition_spec))[:-1]

        data_sequences = utils.partition(self.data, indices[:-1])
        if name is None:
            datasets = [Dataset(data, self.input_features, self.target_features) 
                        for data in data_sequences]
        else:
            datasets = [Dataset(data, self.input_features, 
                                self.target_features, name) 
                        for data, name in zip(data_sequences, names)]

        return datasets

    def convert_to_tensor(self):
        raise NotImplementedError

    def get_statistics(self):
        """
        Returns some basic statistics of the dataset, including 
        the mean, standard deviation, median, upper quartile, lower quartile,
        maximum, and minimum size (in bytes) of a data point. 
        """
        data_point_sizes = [sys.getsizeof(data_point) for data_point in self.data]
        return utils.get_statistics(data_point_sizes)


class TextDataset(Dataset):

    def __repr__(self):
        if self.name is None:
            string = "<TextDataset(size={})>".format(len(self))
        else:
            string = "<TextDataset(name={}, size={})>".format(self.name, len(self))
        return string

    def truncate(self, size=None, tokens=None, in_place=True):

        def counter(element):
                count = 0
                if isinstance(element, str):
                    count += 1
                elif (isinstance(element, list) or isinstance(element, tuple) 
                        or isinstance(element, set)):
                    count += sum(element)
                return count

        if size:
            data = self.data[:size]
        else:
            data = self.data[:count_index(self.data, counter, tokens)]
        if in_place:
            self.data = data
        else:
            return Dataset(data, self.input_features, self.target_features, name)

    def convert_texts(self, curr_encoding="utf-8", new_encoding="utf-8", 
                      error_handling="strict"):
        self.decode_texts(curr_encoding, error_handling)
        self.encode_texts(new_encoding, error_handling)

    def encode_texts(self, encoding="utf-8", error_handling="strict"):
        """
        Args:
            encoding: Str; encoding of the text data (default: 'utf-8'). 
            error_handling: Str; accepts 'strict', 'replace', 'ignore', 
                or 'backslashreplace' (see Python docs for meaning) 
                (default: 'strict').
        """
        @Dataset.datamethod()
        def encode(element):
            if isinstance(element, str):
                new_element = element.encode(encoding, error_handling)
            else:
                new_element = element
            return new_element

        self.map(encode)

    def decode_texts(self, encoding="utf-8", error_handling="strict"):
        """
        Args:
            encoding: Str; encoding of the text data (default: 'utf-8'). 
            error_handling: Str; accepts 'strict', 'replace', 'ignore', 
                or 'backslashreplace' (see Python docs for meaning) 
                (default: 'strict').
        """
        @Dataset.datamethod()
        def decode(element):
            if isinstance(element, bytes):
                new_element = element.decode(encoding, error_handling)
            else:
                new_element = element
            return new_element

        self.map(decode)

    def split_texts(self, split_into=None, split_on=None, splitter=None, 
                    language=None, split_data_points=True, in_place=True):
        """
        Args:
            split_into: Str; accepts either 'sentences', 'lines', 
                or 'paragraphs' (optional, default: None).
            split_on: List; a sequence of strings on which to split.
            splitter: Callable; accepts a string and returns a list of strings.
            language: Str; the 2-letter code or English name of a language, 
                to potentially allow for more accurate splitting; if None 
                and split_into is not None, it falls back on simple universal 
                splitting methods (optional, default: None). 
            split_data_points: Bool; if True, each piece of the split texts 
                becomes a new data point (note that this requires that 
                either the input or the target is empty for every data point), 
                otherwise, if False, there is no splitting of the data points. 
            in_place: Bool.
        """
        if split_data_points and (any(data_point[0] for data_point in self.data) 
                and any(data_point[1] for data_point in self.data)):
            raise NotImplementedError("To split data points, we require either " 
                                      "empty inputs or empty targets.")

        @Dataset.datamethod(multi=True, split=split_data_points)
        def split(element):
            if not isinstance(element, str):
                segments = element
            elif split_into is not None or split_on is not None:
                segments = utils.split_string(
                    element, split_into, split_on, language)
            elif splitter is not None:
                segments = splitter(element)
            return segments

        return self.map(split, multi=split_data_points, in_place=in_place)

    def tokenize(self, token_type, tokenizer=None):
        """
        Args:
            token_type: Str; accepts 'token', 'subtoken', or 'character'.
            tokenizer: Callable; accepts a string and returns a list of tokens;
                if not None, token_type should be 'token' (optional, default: None).
        """
        @Dataset.datamethod()
        def tokenize_element(element):
            if not isinstance(element, str):
                tokens = element
            elif token_type == "character":
                tokens = list(element)
            elif token_type == "token":
                tokens = tokenizer(element)
            elif token_type == "subtoken":
                raise NotImplementedError
            else:
                raise ValueError("Unrecognized token type: '" + token_type + "'.")
            return tokens

        self.map(tokenize_element)

    def clean(self, cleaner=None):
        """
        Args:
            cleaner: Callable; accepts a string and returns the cleaned string.
        """
        @Dataset.datamethod()
        def clean_element(element):
            if not isinstance(element, str):
                cleaned = element
            else:
                cleaned = cleaner(element)
            return cleaned

        self.map(clean_element)

    def correct_spelling(self, spell_corrector):
        """
        Args:
            spell_corrector: Callable or Dict; either a function that accepts
                a string and outputs a spell corrected version of the string
                or a dictionary mapping misspelled strings to their 
                spell-corrected versions. Note: be sure to tokenize the dataset
                first if spell_corrector only accepts tokens.
        """
        if isinstance(spell_corrector, dict):
            spell_corrector = lambda string: spell_corrector[string]

        @Dataset.datamethod()
        def correct(element):
            if not isinstance(element, str):
                corrected = element
            else:
                corrected = spell_corrector(element)
            return corrected

        self.map(correct)

    def generate_vocabulary(self, separate=True):
        """
        Assumes that all text in the dataset is tokenized, 
        i.e. the base text elements in each data point are tokens.

        Args:
            separate: Bool; determines whether separate vocabularies
                for the input data and target data are generated or not.
        """
        input_vocabulary, target_vocabulary = Counter(), Counter()
        for input_data, target_data in self.data:
            utils.apply_to_collection(input_vocabulary.update, input_data)
            utils.apply_to_collection(target_vocabulary.update, target_data)
        if separate:
            return input_vocabulary, target_vocabulary
        else:
            return input_vocabulary + target_vocabulary

    def convert_to_ids(self, vocab_mapping=None, converter=None):
        """
        Assumes that all text in the dataset is tokenized, 
        i.e. the base text elements in each data point are tokens.

        Args:
            vocab_mapping: Dict; a mapping from vocabulary tokens to ids 
                (optional, default: None).
            convert: Callable; applied to every element of every data point, 
                should return a list of ids (optional, default: None).
        """
        multi = vocab_mapping is None
        if vocab_mapping is not None:
            converter = lambda element: vocab_mapping[element]

        @Dataset.datamethod(multi=multi)
        def convert(element):
            return converter(element)

        self.map(convert)

    def get_statistics(self, tokenized=True):

        def char_count(element):
            count = 0
            if isinstance(element, str):
                count += len(element)
            elif (isinstance(element, list) or isinstance(element, tuple) 
                    or isinstance(element, set)):
                count += sum(element)
            return count

        def token_count(element):
            count = 0
            if isinstance(element, str):
                count += 1
            elif (isinstance(element, list) or isinstance(element, tuple) 
                    or isinstance(element, set)):
                count += sum(element)
            return count

        byte_statistics = super().get_statistics()
        character_counts = [
            utils.aggregate_on_collection(char_count, input_data) 
            + utils.aggregate_on_collection(char_count, target_data)
            for input_data, target_data in self.data]
        if tokenized:
            token_counts = [
                utils.aggregate_on_collection(token_count, input_data) 
                + utils.aggregate_on_collection(token_count, target_data)
                for input_data, target_data in self.data]
        statistics = {
            "byte": byte_statistics,
            "character": utils.get_statistics(character_counts)
        }
        if tokenized:
            statistics["token"] = utils.get_statistics(token_counts)

        return statistics


class ImageDataset(Dataset):

    def __repr__(self):
        if self.name is None:
            string = "<ImageDataset(size={})>".format(len(self))
        else:
            string = "<ImageDataset(name={}, size={})>".format(self.name, len(self))
        return string


class AudioDataset(Dataset):

    def __repr__(self):
        if self.name is None:
            string = "<AudioDataset(size={})>".format(len(self))
        else:
            string = "<AudioDataset(name={}, size={})>".format(self.name, len(self))
        return string


class VideoDataset(Dataset):

    def __repr__(self):
        if self.name is None:
            string = "<VideoDataset(size={})>".format(len(self))
        else:
            string = "<VideoDataset(name={}, size={})>".format(self.name, len(self))
        return string


class TensorDataset:

    def __init__(self, input_data, target_data, input_features=None, 
                 target_features=None, name=None):
        """
        Args:
            input_data: List; a sequence of PyTorch Tensors 
                with equivalent sizes along the first dimension.
            target_data: List; a sequence of PyTorch Tensors
                with equivalent sizes along the first dimension 
                (and equivalent to the tensors in input_data as well).
            input_features: Dict; a mapping from indices of input_data 
                to feature names.
            target_features: Dict; a mapping from indices of target_data 
                to feature names.
            name: Str; used for storage and loading (optional, default: None).
        """
        if len({tensor.size(0) for tensor in input_data} 
                | {tensor.size(0) for tensor in target_data}) != 1:
            raise ValueError("Incompatible tensors!")
        self.input_data = input_data
        self.target_data = target_data
        self.input_features = input_features
        self.target_features = target_features
        self.name = name

    def __getitem__(self, index):
        return ([tensor[index] for tensor in self.input_data], 
                [tensor[index] for tensor in self.target_data])

    def __len__(self):
        return self.input_data[0].size(0)

    def __add__(self, other):
        return self.concatenate(other)

    def __repr__(self):
        if self.name is None:
            string = "<TensorDataset(size={})>".format(len(self))
        else:
            string = "<TensorDataset(name={}, size={})>".format(self.name, len(self))
        return string

    @classmethod
    def from_storage(cls, file_path, name=None):
        """
        Args:
            file_path: Str; full path to the file containing the dataset.
            name: Str; name of the dataset (optional, default: None). 
        """
        data = torch.load(file_path)
        return cls(input_data=data[0], target_data=data[1],
                   input_features=data[2], target_features=data[3], name=data[4])

    @classmethod
    def concatenate(cls, *datasets, name=None):
        # Expects each dataset in datasets to have 'matching' input 
        # and target tensors, and matching feature dictionaries.
        input_tensors = [torch.cat([dataset.input_data[i] for dataset in datasets]) 
                         for i in range(len(datasets[0].input_data))]
        target_tensors = [torch.cat([dataset.target_data[i] for dataset in datasets]) 
                         for i in range(len(datasets[0].target_data))]
        return cls(input_data=input_tensors, target_data=target_tensors, 
                   input_features=datasets[0].input_features, 
                   target_features=datasets[0].target_features,
                   name=name)

    def store(self, dir_name, file_name=None):
        """
        Args:
            dir_name: Str; path of a directory where the dataset will be stored.
            file_name: Str; name of the dataset file; if not provided, 
                resorts to the name of the dataset (optional, default: None). 
        """
        if file_name is None:
            file_name = self.name
        torch.save((self.input_data, self.target_data, self.input_features, 
                    self.target_features, self.name), os.path.join(dir_name, file_name))

    def convert_to_native(self):
        return NotImplementedError

    def get_statistics(self):
        return self.input_data[0].size()


class DataSource:

    def __init__(self, tensor_dataset=None, data=None, size_limit=None, 
                 random_access=False, name=None):
        """
        Args:
            tensor_dataset: TensorDataset; (optional, default: None).
            data: List; a sequence of data points of the form (input, target).
            size_limit: Int; maximum number of data points held 
                in the data source (optional, default: None).
            random_access: Bool; determines whether to optimize the data source
                for random access (if True) or sequential access (if False).
        """
        if data is not None and size_limit is not None and len(data) > size_limit:
            raise ValueError("Length of 'data' exceeds size limit!")
        self.size_limit = size_limit
        self._random_access = random_access
        self.name = name
        if tensor_dataset is not None:
            self.initialize_with_dataset(tensor_dataset)
        else:
            replace = lambda element: element if element else None
            data = [(replace(input_data), replace(target_data)) 
                    for input_data, target_data in data]
            if random_access:
                self.data = list(data)
            else:
                self.data = deque(data, size_limit)

        # tracks the last (reverse) index of the last requested batch
        # takes into account updates, only valid if random sampling never used
        self._state_index = 0

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if name is None:
            string = "<DataSource(size={}, size_limit={})>".format(
                len(self), self.size_limit)
        else:
            string = "<DataSource(name={}, size={}, size_limit={})>".format(
                self.name, len(self), self.size_limit)
        return string

    def initialize_with_dataset(self, tensor_dataset):
        dataset = tensor_dataset
        if dataset.input_features is None:
            self.input_data = [dataset[i][0] for i in range(len(dataset))]
        else:
            self.input_data = [{dataset.input_features[j]: dataset[i][0][j] 
                                for j in range(len(dataset[i][0]))} 
                               for i in range(len(dataset))]
        if dataset.target_features is None:
            self.target_data = [dataset[i][1] for i in range(len(dataset))]
        else:
            self.target_data = [{dataset.target_features[j]: dataset[i][1][j] 
                                 for j in range(len(dataset[i][1]))} 
                                for i in range(len(dataset))]

        if self._random_access:
            self.data = list(zip(self.input_data, self.target_data))
        else:
            self.data = deque(zip(self.input_data, self.target_data), self.size_limit)

        replace = lambda element: element if element else None
        self.data = [(replace(input_data), replace(target_data)) 
                     for input_data, target_data in self.data]

        if size_limit is not None and len(self.data) > size_limit:
            raise ValueError("Dataset exceeds size limit!")

        self._state_index = 0

    def load(self, file_path, name=None):
        """
        Args:
            file_path: Str; full path to the file containing the dataset.
            name: Str; name of the dataset (optional, default: None). 
        """
        data = torch.load(file_path)
        return cls(data=data[0], size_limit=data[1], 
                   random_access=data[2], name=data[3])

    def store(self, dir_name, file_name=None):
        """
        Args:
            dir_name: Str; path of a directory where the data source will be stored.
            file_name: Str; name of the data source file; if not provided, 
                resorts to the name of the data source (optional, default: None).
        """
        if file_name is None:
            file_name = self.name
        torch.save((self.data, self.size_limit, self._random_access, self.name), 
                   os.path.join(dir_name, file_name))

    def reset(self):
        self._state_index = 0

    def convert_to_dataset(self):
        raise NotImplementedError

    def shuffle(self):
        random.shuffle(self.data)

    def update(self, data, append=False):
        """
        Adds data to the data source while maintaining any size limit.

        Args:
            data: List or tuple; either a single data point of the form 
                (input, target) or a list of such data points.
            append: Bool; determines whether to append the data to the front/end 
                of the data source or push it to the back/start 
                (optional, default: False).
        """
        if not append and self._random_access:
            warnings.warn("Pushing data with random_access == True " 
                          "could cause significant performance issues"
                          "---try appending instead.", RuntimeWarning)
        if isinstance(data, tuple) and len(data) == 2:
            data = [data]
        if append:
            self.data.extend(data)
            if len(self) > self.size_limit:
                self.data = self.data[len(self)-self.size_limit:] 
        else:
            if isinstance(self.data, list):
                self.data = data + self.data
            else:
                data.reverse()
                self.data.extendleft(data)
            if len(self) > self.size_limit:
                if self._state_index >= len(self) - self.size_limit:
                    self._state_index -= len(self) - self.size_limit
                self.data = self.data[:self.size_limit]

    def get_next_batch(self, batch_size, random_sample=False, 
                       with_replacement=True, concat_batchwise=False):
        """
        Args:
            batch_size: Int.
            random_sample: Bool; whether to randomly sample the batch 
                from the data source (optional, default: False).
            with_replacement: Bool; whether to sample the batch from 
                the data source with replacement or not (optional, default: True).
            concat_batchwise: Bool; whether to return the batch with 
                all data points concatenated together batchwise into 
                a single Tensor for each feature; requires the data points 
                to either contain lists of Tensors or dictionaries 
                with Tensor values (optional, default: False).
        Returns:
            Batch of data points and a boolean indicating whether the data source 
            has been exhausted or not (that is, whether the entire data source 
            has been used and its state now reset). If concat_batchwise is True,
            then it also returns the original non-concatenated sequence of data points. 
        """
        data_source_exhausted = False
        if random_sample and not self._random_access:
            warnings.warn("Randomly sampling batches with random_access == False " 
                          "could cause significant performance issues.", 
                          RuntimeWarning)
        if random_sample and not with_replacement:
            warnings.warn("Randomly sampling batches without replacement " 
                          "could cause significant performance issues.", 
                          RuntimeWarning)

        if random_sample:
            if with_replacement:
                batch_seq = random.sample(self.data, batch_size)
            else:
                batch_seq, self.data = utils.sample_without_replace(
                    self.data, batch_size)
        else:
            if with_replacement:
                batch_seq = [self.data[-i-1-self._state_index] 
                             for i in range(batch_size)]
                self._state_index += batch_size
                if self._state_index >= len(self):
                    self._state_index = 0
                    data_source_exhausted = True
            else:
                batch_seq = [self.data.pop() for i in range(batch_size)]

        if concat_batchwise:
            # Assumes that data points either contain (compatible) Tensors 
            # or (compatible) dictionaries with Tensor values
            if hasattr(batch_seq[0][1], "storage"):  # check if Pytorch Tensor
                input_data = sorted([data_point[0] for data_point in batch_seq 
                                    if data_point[0] is not None], 
                                    key=lambda x: x.shape[0])
                target_data = sorted([data_point[1] for data_point in batch_seq
                                      if data_point[0] is not None], 
                                     key=lambda x: x.shape[0])
                batch = (
                    tensor_utils.pad_sequence(
                        input_data, batch_first=True) if input_data else None, 
                    tensor_utils.pad_sequence(
                        target_data, batch_first=True) if target_data else None,
                )
            elif isinstance(batch_seq[0][1], dict):
                if all(data_point[0] is None for data_point in batch_seq):
                    input_data = None
                else:
                    input_data = {
                        key: sorted([data_point[0][key] for data_point in batch_seq], 
                                    key=lambda x: x.shape[0])
                        for key in batch_seq[0][0]
                    }
                if all(data_point[1] is None for data_point in batch_seq):
                    target_data = None
                else:
                    target_data = {
                        key: sorted([data_point[1][key] for data_point in batch_seq], 
                                    key=lambda x: x.shape[0])
                        for key in batch_seq[0][1]
                    }
                batch = (
                    {key: tensor_utils.pad_sequence(value, batch_first=True) 
                     for key, value in input_data.items()} if input_data else None, 
                    {key: tensor_utils.pad_sequence(value, batch_first=True) 
                     for key, value in target_data.items()} if target_data else None,
                )
            else:
                raise ValueError("Don't know how to concatenate batches " 
                                 "with data points of this form.")
        else:
            batch = batch_seq

        if concat_batchwise:
            return batch, batch_seq, data_source_exhausted
        else:
            return batch, data_source_exhausted

    def get_feedback(self, output_data):
        """
        A method the user should override to dynamically return feedback 
        for learning---e.g. returning a reward for reinforcement learning 
        (in this case, output_data would likely be actions). 
        """
        raise NotImplementedError

    def get_statistics(self):
        raise NotImplementedError

