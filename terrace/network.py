""""""

import os
import json
from datetime import datetime
from ast import literal_eval
from pprint import pformat

import numpy as np
import torch
from torch import nn
import torch.nn.functional as func
from torch.autograd import Variable as TorchVariable

from . import data_utilities as utils


CUDA_AVAILABLE = torch.cuda.is_available()


class Variable:

    def __new__(cls, *args, use_cuda=True, device=None, **kwargs):
        if use_cuda:
            return TorchVariable(*args, **kwargs).cuda(device=device)
        else:
            return TorchVariable(*args, **kwargs)


class HParameters:

    # Similar to tf.contrib.training.HParams

    def __init__(self, **kwargs):
        self._hparam_types = {}
        self.update(kwargs)

    def __repr__(self):
        return "Hparameters(" + pformat(self.values()) + ")"

    def update(self, values):
        for name, value in values.items():
            if hasattr(self, name):
                if isinstance(value, self._hparam_types[name]):
                    setattr(self, name, value)
                else:
                    raise ValueError("Hyperparameter '%s' requires type '%s', "
                                     "while input value has type '%s'." 
                                     % (name, self._hparam_types[name], type(value)))
            else:
                setattr(self, name, value)
                self._hparam_types[name] = type(value)

    def values(self):
        return {name: getattr(self, name) for name in self._hparam_types.keys()}

    def save(self, dir_path, file_name):
        hparams = self.values()
        utils.save(json.dumps(hparams), os.path.join(dir_path, file_name))

    def load(self, dir_path, file_name):
        with open(os.path.join(dir_path, file_name), "r") as hparam_file:
            hparams = literal_eval(hparam_file.read())
        self.update(hparams)


def find_latest_model(dir_path, parameters_only=True):
    model_files = [file_name for file_name in os.listdir() 
                   if os.path.isfile(os.path.join(dir_path, file_name)) 
                   and ((file_name.endswith(".params") and parameters_only)
                   or (file_name.endswith(".object") and not parameters_only))]
    return sorted(model_files)[-1]


class Module(nn.Module):

    @classmethod
    def from_storage(self, dir_path, file_name=None):
        """
        Loads module object from file---may produce issues if this class 
        or Python module has been modified since the file was created.
        """
        if file_name is None:
            file_name = find_latest_model(dir_path, parameters_only=False)
        return torch.load(os.path.join(dir_path, file_name))

    def save(self, dir_path, file_name=None, parameters_only=True):
        if file_name is None:
            file_name = "model-" + utils.formatted_time()
            file_name += ".params" if parameters_only else ".object"
        file_path = os.path.join(dir_path, file_name)
        if parameters_only:
            torch.save(self.state_dict(), file_path)
        else:
            torch.save(self, file_path)

    def load(self, dir_path, file_name=None):
        """
        Loads parameters on disk for this model; if file_name is None,
        loads data from the latest parameter file in dir_path.
        """
        if file_name is None:
            file_name = find_latest_model(dir_path)
        self.load_state_dict(torch.load(os.path.join(dir_path, file_name)))

    def get_info(self):
        info = {}
        for name, module in self.named_modules():
            # Only include base modules, with no children
            if len(list(module.named_modules())) == 1:
                module_param_info = {
                    name: (list(param.size()), np.prod(param.size())) 
                    for name, param in module.named_parameters()
                }
                info[name] = {"module": str(module), 
                              "parameters": module_param_info}
        if hasattr(self, "hparams"):
            info["hparams"] = self.hparams
        elif hasattr(self, "hyperparams"):
            info["hparams"] = self.hyperparams
        elif hasattr(self, "hparameters"):
            info["hparams"] = self.hparameters
        elif hasattr(self, "hyperparameters"):
            info["hparams"] = self.hyperparameters

        return info

    def get_module_info(self):
        return {name: str(module) for name, module in self.named_modules()}

    def get_parameter_info(self):
        return {name: (list(param.size()), np.prod(param.size())) 
                for name, param in self.named_parameters()}


class ResidualConnections(Module):

    pass


class Attention(Module):

    pass


class Memory(Module):

    pass


class Autoencoder(Module):

    pass


class VAE(Autoencoder):

    pass


class GAN(Module):

    pass