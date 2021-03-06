""""""

import os
import sys
import re
import logging
import functools
from datetime import datetime
from time import time

import numpy as np
from torch import nn
from torch import optim

from . import data_utilities as utils
from . import evaluate


BUILTIN_OPTIMIZERS = {
    "sgd": optim.SGD,
    "asgd": optim.ASGD,
    "rms": optim.RMSprop,
    "adagrad": optim.Adagrad,
    "adadelta": optim.Adadelta,
    "adam": optim.Adam,
    "sparseadam": optim.SparseAdam,
    "adamax": optim.Adamax,
    "amsgrad": functools.partial(optim.Adam, amsgrad=True),
    "lbfgs": optim.LBFGS,
    "rprop": optim.Rprop,
}


BUILTIN_LOSSES = {
    "l1": nn.L1Loss,
    "mse": nn.MSELoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "nll": nn.NLLLoss,
    "poisson_nll": nn.PoissonNLLLoss,
    "kl_div": nn.KLDivLoss,
    "bce": nn.BCELoss,
    "bce_logits": nn.BCEWithLogitsLoss,
    "margin_ranking": nn.MarginRankingLoss,
    "hinge_embedding": nn.HingeEmbeddingLoss,
    "multi_label_margin": nn.MultiLabelMarginLoss,
    "smooth_l1": nn.SmoothL1Loss,
    "soft_margin": nn.SoftMarginLoss,
    "multi_label_soft_margin": nn.MultiLabelSoftMarginLoss,
    "cosine_embedding": nn.CosineEmbeddingLoss,
    "multi_margin": nn.MultiMarginLoss,
}


def setup_logging(root_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(root_dir, "events.log"))
    file_handler.setLevel(logging.INFO)
    print_handler = logging.StreamHandler(sys.stdout)
    print_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    print_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(print_handler)

    return logger


class BaseTrainer:

    def __init__(self, model, hparams, train_data_source, train_step_fn, 
                 callbacks=None, device="gpu", loss_function=None,
                 optimizer_creator=None, eval_data_source=None, logger=None, 
                 training_dir=None):
        """
        Args:
            model: Module.
            hparams: Hparameters.
            train_data_source: DataSource.
            train_step_fn: Callback; called repeatedly, used to execute 
                one training step; should accept as input model, hparams, 
                data_source, loss_function, optimizer, and training_log, as well as 
                any number of additional arguments, and return either None 
                or a dictionary of training data, e.g. {"loss": <float>, ...}.
            callbacks: List; set of TrainingCallbacks that are called 
                at the start of training, before each call of train_step_fn, 
                after each call of train_step_fn, and at the end/shutdown of training
                (optional, default: None).
            device: Str; determines which device is used to train the model, 
                accepts 'gpu' or 'cpu' (optional, default: 'gpu').
            optimizer_creator: Callable; should accepts model and hparams and 
                return an instance of a subclass of torch.optim.Optimizer; if None, 
                searches for optimizer via hparams and built-in optimizers
                (optional, default: None).
            loss_function: Callable; if None, searches for loss function via 
                hparams and built-in loss functions (optional, default: None).
            eval_data_source: DataSource.
            logger: logging.Logger; (optional, default: None).
            training_dir: Str; directory where all training and model data 
                will be stored (optional, default: None).
        """
        self.model = model
        self.hparams = hparams
        self.train_data_source = train_data_source
        self.train_step_fn = train_step_fn
        self.callbacks = callbacks
        self.device = device.lower()
        self.eval_data_source = eval_data_source
        self.logger = logger
        self.training_dir = training_dir
        if self.device == "gpu":
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.loss_function = loss_function or create_loss_function(hparams)
        self.optimizer = (create_optimizer(model, hparams) 
                          if optimizer_creator is None 
                          else optimizer_creator(model, hparams))

        self.training_log = None
        self.callback_log = None
        self._stop_training = False

    def run(self, steps, *args, logging=True, **kwargs):
        """
        Args:
            steps: Int; number of training steps at which, if training has not 
                already stopped, training stops.
            args: Optional positional arguments passed to self.train_step_fn.
            kwargs: Optional keyword arguments passed to self.train_step_fn.
        """
        raise NotImplementedError

    def log(self, message, warning=False):
        if self.logger is not None:
            self.logger.warn(message) if warning else self.logger.info(message)


class Trainer(BaseTrainer):

    def run(self, steps, *args, **kwargs):
        self.model.train()
        self.training_log = []
        self.callback_log = []
        for callback in self.callbacks:
            callback.begin(self)
        for step in range(steps):
            if self._stop_training:
                break
            for callback in self.callbacks:
                callback.before_train_step(self, step)
            if self._stop_training:
                break
            self.model.train()
            log_data = self.train_step_fn(
                self.model, self.hparams, self.train_data_source, 
                self.loss_function, self.optimizer, 
                self.training_log, *args, **kwargs)
            self.training_log.append((str(datetime.utcnow()), step, log_data))
            self.callback_log.append({})
            for callback in self.callbacks:
                output = callback.after_train_step(self, step)
                if output is not None:
                    self.callback_log[-1].update(output)
        for callback in self.callbacks:
            callback.end(self)
            if output is not None:
                self.callback_log[-1].update(output)


class TrainingCallback:
    """
    Abstract base class for training callbacks. 
    """

    def __init__(self):
        pass

    def begin(self, trainer):
        pass

    def before_train_step(self, trainer, step):
        pass

    def after_train_step(self, trainer, step):
        """
        Only data returned by this method will be stored in the callback log.
        Should return a dictionary, where each key is the name of data 
        held in the associated value, e.g. {"eval_loss": <float>, ...}.
        """
        pass

    def end(self, trainer):
        pass


class ReturnException(Exception):
    """
    Exception raised to indicate a training callback method 
    should return immediately.
    """


class PeriodicCallback(TrainingCallback):
    """Only runs once every period steps."""

    def __init__(self, period, initial_run=False, final_run=True, 
                 use_external_clock=True):
        """
        Args:
            period: Int; determines how often the callback is called.
            initial_run: Bool; determines whether the callback is called 
                at the start of training (optional, default: False).
            final_run: Bool; determines whether the callback is called
                at the end of training (optional, default: True).
            use_external_clock: Bool; detrmines whether to use the training step
                input to before_train_step and after_train_step to determine
                when to initiate the callback (if True) or whether to use an 
                internal 'clock' (if False) (optional, default: True).
        """
        super().__init__()
        self.period = period
        self.initial_run = initial_run
        self.final_run = final_run
        self.use_external_clock = use_external_clock
        self.internal_clock = 0

    def begin(self, trainer):
        self.internal_clock = 0
        if not self.initial_run:
            raise ReturnException

    def before_train_step(self, trainer, step):
        if self.use_external_clock and step % self.period:
            raise ReturnException
        elif not self.use_external_clock and self.internal_clock % self.period:
            raise ReturnException 

    def after_train_step(self, trainer, step):
        if self.use_external_clock and step % self.period:
            raise ReturnException
        elif not self.use_external_clock and self.internal_clock % self.period:
            self.internal_clock += 1
            raise ReturnException

    def end(self, trainer):
        if not self.final_run:
            raise ReturnException


class EarlyStopCallback(PeriodicCallback):
    """
    Stops training when stopping_predicate returns True or metric 
    has stopped improving after period*max_wait_time steps 
    (which is assumed to correspond with max_wait_time evaluations of metric).

    Note that if stopping_predicate is not None then only metric 
    and arguments inherited from PeriodicCallback are used. 
    """

    def __init__(self, *args, metric="eval_loss", min_delta=0, 
                 max_wait_time=5, mode="auto", 
                 stopping_predicate=None, **kwargs):
        """
        Args:
            metric: Str; the name of a metric used for determining 
                when to stop; if None, then both the training log and callback log 
                are passed to the stopping predicate 
                (optional, default: 'eval_loss').
            min_delta: Float; minimum change in the monitored metric that qualifies 
                as an improvement (optional, default: 0).
            max_wait_time: Int; number of periods of no improvement after which 
                training will be stopped (optional, default: 5). 
            mode: Str; expects 'min', 'max', or 'auto'---if 'min', then 
                no improvement means no decrease; if 'max', then no improvement 
                means no increase; if 'auto', then the mode is inferred 
                from the value of metric.
            stopping_predicate: Callable; accepts a training log 
                and callback log (if metric is None) or a metric value (otherwise)
                and returns a boolean indicate whether to stop training or not.  
        """
        super().__init__(*args, **kwargs)
        self.metric = metric
        self.min_delta = min_delta
        self.max_wait_time = max_wait_time

        if metric is not None and (mode == "auto" or mode not in ["min", "max"]):
            if "acc" in metric:
                self.mode = "max"
            else:
                self.mode = "min"
        else:
            self.mode = mode

        self.stopping_predicate = stopping_predicate or self._create_predicate()
        self.wait = 0

    def begin(self, trainer):
        try:
            super().begin(trainer)
        except ReturnException:
            return
        self.wait = 0
        self.best_value = np.Inf if self.mode == "min" else -np.Inf

    def _create_predicate(self):

        if self.metric is None:
            raise ValueError("Must provide a stopping predicate if metric is None.")

        def predicate(current_value):
            stop = False
            if self.mode == "min" and self.best_value < current_value + min_delta:
                stop = self.wait >= self.max_wait_time
                self.wait += 1
            elif self.mode == "max" and self.best_value > current_value - min_delta:
                stop = self.wait >= self.max_wait_time
                self.wait += 1
            else:
                self.best_value = current_value
                self.wait = 0

            return stop

        return predicate

    def before_train_step(self, trainer, step):
        try:
            super().before_train_step(trainer, step)
        except ReturnException:
            return

    def after_train_step(self, trainer, step):
        try:
            super().after_train_step(trainer, step)
        except ReturnException:
            return
        if self.metric in trainer.callback_log[step]:
            stopping_data = trainer.callback_log[step][self.metric]
        elif self.metric in trainer.training_log[step][-1]:
            stopping_data = trainer.training_log[step][-1][self.metric]
        elif self.metric is None:
            stopping_data = (trainer.callback_log, trainer.training_log)
        else:
            metrics = (list(trainer.callback_log[step].keys()) 
                       + list(trainer.training_log[step][-1].keys()))
            trainer.log(
                "Early stopping conditioned on an unavailable metric '%s'. " 
                "Available metrics: %s" % (self.metric, ",".join(metrics)), 
                warning=True)
            return

        if isinstance(stopping_data, tuple):
            stop = self.stopping_predicate(*stopping_data)
        else:
            stop = self.stopping_predicate(stopping_data)

        trainer._stop_training = stop
        trainer.log("Early stop condition reached; " 
                    "stopping training at step %s..." % step)

    def end(self, trainer):
        try:
            super().end(trainer)
        except ReturnException:
            return


class TrainingLogCallback(PeriodicCallback):
    """
    Saves the hyperparameters and, periodically, all training and callback data.
    Also, outputs training data to the log, including the time to perform 
    the last training step. 
    """

    def __init__(self, *args, log_file_name="training.log", 
                 append=False, **kwargs):
        """
        Args:
            log_file_name: Str; (optional, default: 'training.log').
            append: Bool; If True, append if file exists; otherwise, overwrite 
                any existing file (optional, default: False).  
        """
        super().__init__(*args, **kwargs)
        self.log_file_name = log_file_name
        self.append = append
        self.initial_run = True
        self.last_step_time = time()

    def begin(self, trainer):
        try:
            super().begin(trainer)
        except ReturnException:
            return
        trainer.hparams.save(trainer.training_dir, "hparams.json")
        if not self.append:
            utils.save("", self.log_file_name)
        self.last_step_time = time()

    def before_train_step(self, trainer, step):
        try:
            super().before_train_step(trainer, step)
        except ReturnException:
            return

    def after_train_step(self, trainer, step):
        try:
            super().after_train_step(trainer, step)
        except ReturnException:
            return
        # Output training loss + timing data
        losses = [(key, value) for key, value in trainer.training_log[-1][-1].items()
                  if "loss" in key]
        losses.sort()
        trainer.log("steps/sec: " + str(round(
            self.period / (time() - self.last_step_time), 4)))
        self.last_step_time = time()
        trainer.log("step %s - " % step + ", ".join(
            [key + " = " + str(round(value, 4)) for key, value in losses]))
        map(lambda handler: handler.flush(), trainer.logger.handlers)
        # Save logs
        log_data = (trainer.training_log[-1], trainer.callback_log[-1])
        log_string = str(log_data) + "\n"
        utils.save(log_string, self.log_file_name, append=True)

    def end(self, trainer):
        try:
            super().end(trainer)
        except ReturnException:
            return


class EvaluationCallback(PeriodicCallback):

    def __init__(self, metrics, *args, eval_function=evaluate.evaluate, 
                 batch_size=None, steps=100, full_final_eval=True, 
                 generate_function=None, num_examples=10, log_examples=False, 
                 **kwargs):
        """
        Args:
            metrics: List; a sequence of strings or callables 
                representing/implementing evaluation metrics; any callable 
                should accept two tensors and return a scalar.
            eval_function: Callable; accepts model, hparams, data_source, 
                metrics, batch_size, and (as a keyword arg) steps and returns 
                a dictionary mapping metrics to values 
                (optional, default: evaluate.evaluate).
            batch_size: Int; if not None, batch size used during 
                evaluation (optional, default: None). 
            steps: Int; steps of evaluation per call to eval_function 
                (optional, default: 100).
            full_final_eval: Bool; indicates whether an evaluation on the
                entire validation data source should be performed at the end
                of training (optional, default: True).
            generate_function: Callable; accepts model, hparams, data_source, 
                and (as a keyword arg) num_examples and returns a list 
                of num_examples 2-tuples of the form (input, generated_output);
                note that if log_examples is False, this callable should handle
                the storage/output of all generated examples 
                (optional, default: None).
            num_examples: Int; passed to generate_function, should determine 
                the number of examples processed/generated by the model 
                (optional, default: 10).
            log_examples: Bool; determines whether to log examples returned 
                by generate_function (optional, default: False). 
        """
        super().__init__(*args, **kwargs)
        self.metrics = metrics
        self.eval_function = eval_function
        self.batch_size = batch_size
        self.steps = steps
        self.full_final_eval = full_final_eval
        self.generate_function = functools.partial(
            generate_function, num_examples=num_examples)
        self.log_examples = log_examples

        self.initial_run = True

    def begin(self, trainer):
        try:
            super().begin(trainer)
        except ReturnException:
            return
        if self.batch_size is None:
            self.batch_size = trainer.hparams.batch_size

    def _evaluate_and_log(self, model, hparams, 
                          data_source, log_function, step=None):
        if step is None and self.full_final_eval:
            steps = None
        else:
            steps = self.steps
        if step is None:
            step = "final"
        results = self.eval_function(model, hparams, data_source, 
                                     self.metrics, self.batch_size, steps=steps)
        log_function("evaluation - step %s - " % step + ", ".join(
            [str(metric) + " = " + str(value) for metric, value in results.items()]))
        if self.generate_function is not None:
            examples = self.generate_function(model, hparams, data_source)
            if self.log_examples:
                has_input = any(example[0] for example in examples)
                log_function("\n" + "\n".join(
                    [" --- input: " + str(input_ex) + "\n     output: " + str(output_ex)
                     if has_input else " --- output: " + str(output_ex)
                     for input_ex, output_ex in examples]))

        return results

    def before_train_step(self, trainer, step):
        try:
            super().before_train_step(trainer, step)
        except ReturnException:
            return

    def after_train_step(self, trainer, step):
        try:
            super().after_train_step(trainer, step)
        except ReturnException:
            return
        if trainer.eval_data_source is None:
            trainer.log("Validation data source not provided; "
                        "unable to evaluate model.", warning=True)
        else:
            return self._evaluate_and_log(
                trainer.model, trainer.hparams, trainer.eval_data_source, 
                trainer.log, step)

    def end(self, trainer):
        try:
            super().end(trainer)
        except ReturnException:
            return
        if trainer.eval_data_source is None:
            trainer.log("Validation data source not provided; "
                        "unable to evaluate model.", warning=True)
        else:
            return self._evaluate_and_log(
                trainer.model, trainer.hparams, 
                trainer.eval_data_source, trainer.log)


class SaverCallback(PeriodicCallback):
    """Saves the model periodically."""

    def __init__(self, *args, model_file_prefix=None, 
                 parameters_only=True, max_stored=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_file_prefix = (model_file_prefix 
                                  if model_file_prefix is not None else "model")
        self.parameters_only = parameters_only
        self.max_stored = max_stored
        self.model_file_regex = r".*\\" + self.model_file_prefix + r"-\d+"
        self.model_file_regex += r"\.params" if self.parameters_only else r"\.object"
        self.model_file_regex = re.compile(self.model_file_regex)

    def _save_model(self, model, root_dir, step):
        file_name = self.model_file_prefix + "-" + str(step)
        file_name += ".params" if self.parameters_only else ".object"
        model.save(root_dir, file_name, self.parameters_only)

    def begin(self, trainer):
        try:
            super().begin(trainer)
        except ReturnException:
            return
        self._save_model(trainer.model, trainer.training_dir, 0)
        trainer.log("Model saved in %s." % trainer.training_dir)

    def before_train_step(self, trainer, step):
        try:
            super().before_train_step(trainer, step)
        except ReturnException:
            return

    def after_train_step(self, trainer, step):
        try:
            super().after_train_step(trainer, step)
        except ReturnException:
            return
        self._save_model(trainer.model, trainer.training_dir, step)
        trainer.log("Model saved in %s." % trainer.training_dir)
        if self.max_stored is not None:
            stored_models = utils.find_filenames(
                trainer.training_dir, self.model_file_regex, walk=False)
            if len(stored_models) > self.max_stored:
                stored_models.sort()
                utils.delete_files(stored_models[:-self.max_stored])

    def end(self, trainer):
        try:
            super().end(trainer)
        except ReturnException:
            return
        self._save_model(trainer.model, trainer.training_dir, "final")
        trainer.log("Model saved in %s." % trainer.training_dir)


class HyperparameterScheduleCallback(PeriodicCallback):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError

    def begin(self, trainer):
        try:
            super().begin(trainer)
        except ReturnException:
            return

    def before_train_step(self, trainer, step):
        try:
            super().before_train_step(trainer, step)
        except ReturnException:
            return

    def after_train_step(self, trainer, step):
        try:
            super().after_train_step(trainer, step)
        except ReturnException:
            return

    def end(self, trainer):
        try:
            super().end(trainer)
        except ReturnException:
            return


def get_optimizer_parameters(parameter_names, hparams):
    param_names = ["optimizer_" + param for param in parameter_names]
    param_dict = utils.filter_dict(hparams.values(), param_names)
    return {param[10:]: value for param, value in param_dict.items()}


def create_optimizer(model, hparams=None, optimizer_class=None, **kwargs):
    """
    Expects all optimizer hyperparameters to have names 
    starting with 'optimizer_'. Also, expects all optimizer information
    to be in either hparams or optimizer_class and kwargs.

    Args:
        model: Module.
        hparams: Hparameters; (optional, default: None).
        optimizer_class: A subclass of torch.optim.Optimizer; 
            (optional, default: None).
        kwargs: Keyword arguments passed to optimizer_class 
            if optimizer_class is not None.
    """ 
    if optimizer_class is None:
        if hparams.optimizer.lower() not in BUILTIN_OPTIMIZERS:
            raise ValueError("Unable to determine which optimizer to use.")
        else:
            optimizer_class = BUILTIN_OPTIMIZERS[hparams.optimizer.lower()]
        try:
            learning_rate = hparams.optimizer_lr
        except AttributeError:
            learning_rate = None
        if optimizer_class == optim.SGD:
            param_args = get_optimizer_parameters(
                ["momentum", "dampening", "weight_decay", "nesterov"], hparams)
        elif optimizer_class == optim.ASGD:
            param_args = get_optimizer_parameters(
                ["lambd", "alpha", "t0", "weight_decay"], hparams)
        elif optimizer_class == optim.RMSprop:
            param_args = get_optimizer_parameters(
                ["alpha", "eps", "weight_decay", "momentum", "centered"], hparams)
        elif optimizer_class == optim.Adagrad:
            param_args = get_optimizer_parameters(
                ["lr_decay", "weight_decay"], hparams)
        elif optimizer_class == optim.Adadelta:
            param_args = get_optimizer_parameters(
                ["rho", "eps", "weight_decay"], hparams)
        elif optimizer_class == optim.Adam or (
                isinstance(optimizer_class, functools.partial) 
                and optimizer_class.func == optim.Adam):
            param_args = get_optimizer_parameters(
                ["betas", "eps", "weight_decay"], hparams)
        elif optimizer_class == optim.SparseAdam:
            param_args = get_optimizer_parameters(
                ["betas", "eps"], hparams)
        elif optimizer_class == optim.Adamax:
            param_args = get_optimizer_parameters(
                ["betas", "eps", "weight_decay"], hparams)
        elif optimizer_class == optim.LBFGS:
            param_args = get_optimizer_parameters(
                ["max_iter", "max_eval", "tolerance_grad", "tolerance_change", 
                 "history_size", "line_search_fn"], hparams)
        elif optimizer_class == optim.Rprop:
            param_args = get_optimizer_parameters(
                ["etas", "step_sizes"], hparams) 
        if learning_rate is None:
            optimizer = optimizer_class(model.parameters(), **param_args)
        else:
            optimizer = optimizer_class(model.parameters(), lr=learning_rate,
                                        **param_args)
    else:
        optimizer = optimizer_class(**kwargs)

    return optimizer


def get_loss_parameters(hparams):
    values = hparams.values()
    param_names = [value for value in values if value.startswith("loss_")]
    param_dict = utils.filter_dict(hparams.values(), param_names)
    return {param[5:]: value for param, value in param_dict.items()}

def create_loss_function(hparams):
    loss_kwargs = get_loss_parameters(hparams)
    if hparams.loss.lower() not in BUILTIN_LOSSES:
        raise ValueError("Unable to determine which loss function to use.")
    loss_module = BUILTIN_LOSSES[hparams.loss.lower()]
    return loss_module(**loss_kwargs)