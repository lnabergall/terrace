""""""

import os
import re
import logging
from datetime import datetime
from time import time

import numpy as np
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
    "lbfgs": optim.LBFGS,
    "rprop": optim.Rprop,
}


def setup_logging(root_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(os.path.join(root_dir, "events.log"))
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

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
                data_source, loss_function, and optimizer, as well as 
                any number of additional arguments, and return either None 
                or a dictionary of training data, e.g. {"loss": <float>, ...}.
            callbacks: List; set of TrainingCallbacks that are called 
                at the start of training, before each call of train_step_fn, 
                after each call of train_step_fn, and at the end/shutdown of training
                (optional, default: None).
            device: Str; determines which device is used to train the model, 
                accepts 'gpu' or 'cpu' (optional, default: 'gpu').
            optimizer_creator: Callable; should accepts model and hparams and 
                return an instance of a subclass of torch.optim.Optimizer; 
                (optional, default: None).
            loss_function: Callable; (optional, default: None).
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
            model.cuda()
        else:
            model.cpu()
        self.loss_function = loss_function or create_loss_function(hparams, model)
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
            log_data = self.train_step_fn(
                self.model, self.hparams, self.data_source, 
                self.loss_function, self.optimizer, *args, **kwargs)
            self.training_log.append((datetime.utcnow(), log_data))
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


def PeriodicCallback(TrainingCallback):
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
        self.period = period
        self.initial_run = initial_run
        self.final_run = final_run
        self.use_external_clock = use_external_clock
        self.internal_clock = 0

    def begin(self, trainer):
        self.internal_clock = 0
        if not self.initial_run:
            return

    def before_train_step(self, trainer, step):
        if self.use_external_clock and step % period:
            return
        elif not self.use_external_clock and self.internal_clock % period:
            return

    def after_train_step(self, trainer, step):
        if self.use_external_clock and step % period:
            return
        elif not self.use_external_clock and self.internal_clock % period:
            return
        self.internal_clock += 1

    def end(self, trainer):
        if not self.final_run:
            return


def EarlyStopCallback(PeriodicCallback):
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
            metric: Str or None; the name of a metric used for determining 
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
        super().begin(trainer)
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

    def after_train_step(self, trainer, step):
        super().after_train_step(trainer, step)
        if self.metric in trainer.callback_log[step]:
            stopping_data = trainer.callback_log[step][self.metric]
        elif self.metric in trainer.training_log[step][1]:
            stopping_data = trainer.training_log[step][1][self.metric]
        elif self.metric is None:
            stopping_data = (trainer.callback_log, trainer.training_log)
        else:
            metrics = (list(trainer.callback_log[step].keys()) 
                       + list(trainer.training_log[step][1].keys()))
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


def TrainingLogCallback(PeriodicCallback):
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
        super().begin(trainer)
        trainer.hparams.save(trainer.training_dir, "hparams.json")
        if not self.append:
            utils.save("", self.log_file_name)
        self.last_step_time = time()

    def after_train_step(self, trainer, step):
        super().after_train_step(trainer, step)
        # Output training loss + timing data
        losses = [(key, value) for key, value in trainer.training_log[-1][1].items()
                  if "loss" in key]
        losses.sort()
        trainer.log("steps/sec: " + str((time() - self.last_step_time) / self.period))
        self.last_step_time = time()
        trainer.log("step %s - " + ", ".join(
            [key + " = " + str(value) for key, value in losses]) % step)
        # Save logs
        log_data = (trainer.training_log[-1], trainer.callback_log[-1])
        log_string = "\n" + str(log_data)
        utils.save(log_string, self.log_file_name, append=True)


def EvaluationCallback(PeriodicCallback):

    def __init__(self, metrics, *args, eval_function=evaluate.evaluate, 
                 batch_size=None, steps=100, full_final_eval=True, **kwargs):
        """
        Args:
            metrics: List; a sequence of strings or callables 
                representing/implementing evaluation metrics; any callable 
                should accept two tensors and return a scalar.
            eval_function: Callable; accepts model, hparams, data_source, 
                metrics, batch_size, and steps and returns a dictionary mapping 
                metrics to values (optional, default: evaluate.evaluate).
            batch_size: Int or None; if not None, batch size used during 
                evaluation (optional, default: None). 
            steps: Int; steps of evaluation per call to eval_function 
                (optional, default: 100).
            full_final_eval: Bool; indicates whether an evaluation on the
                entire validation data source should be performed at the end
                of training (optional, default: True). 
        """
        super().__init__(*args, **kwargs)
        self.metrics = metrics
        self.eval_function = eval_function
        self.batch_size = batch_size
        self.steps = steps
        self.full_final_eval = full_final_eval

    def _evaluate_and_log(self, model, hparams, 
                          data_source, log_function, step=None):
        if step is None and self.full_final_eval:
            steps = None
        else:
            steps = self.steps
        results = self.eval_function(model, hparams, data_source, 
                                     self.metrics, self.batch_size, steps)
        log_function("evaluation - step %s - " + ", ".join(
            [str(metric) + " = " + str(value) for metric, value in results.items()]))
        return results

    def after_train_step(self, trainer, step):
        super().after_train_step(trainer, step)
        if trainer.eval_data_source is None:
            trainer.log("Validation data source not provided; "
                        "unable to evaluate model.", warning=True)
        else:
            return self._evaluate_and_log(
                trainer.model, trainer.hparams, trainer.eval_data_source, 
                trainer.log, step)

    def end(self, trainer):
        super().end(trainer)
        if trainer.eval_data_source is None:
            trainer.log("Validation data source not provided; "
                        "unable to evaluate model.", warning=True)
        else:
            return self._evaluate_and_log(
                trainer.model, trainer.hparams, 
                trainer.eval_data_source, trainer.log)


def SaverCallback(PeriodicCallback):
    """Saves the model periodically."""

    def __init__(self, *args, model_file_prefix=None, 
                 parameters_only=True, max_stored=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_file_prefix = model_file_prefix
        self.parameters_only = parameters_only
        self.max_stored = max_stored
        self.model_file_regex = r".*\\" + self.model_file_prefix + r"-\d+"
        self.model_file_regex += r"\.params" if self.parameters_only else r"\.object"
        self.model_file_regex = re.compile(self.model_file_regex)

    def _save_model(self, model, root_dir, step):
        file_name = self.model_file_prefix + "-" + str(step)
        file_name += ".params" if self.parameters_only else ".object"
        model.save(root_dir, file_name, parameters_only)

    def begin(self, trainer):
        super().begin(trainer)
        self._save_model(trainer.model, trainer.training_dir, 0)
        trainer.log("Model saved in %s." % os.path.split(trainer.training_dir)[1])

    def after_train_step(self, trainer, step):
        super().after_train_step(trainer, step)
        self._save_model(trainer.model, trainer.training_dir, step)
        trainer.log("Model saved in %s." % os.path.split(trainer.training_dir)[1])
        if self.max_stored is not None:
            stored_models = utils.find_filenames(
                trainer.training_dir, self.model_file_regex, walk=False)
            if len(stored_models) > self.max_stored:
                stored_models.sort()
                utils.delete_files(stored_models[:-self.max_stored])

    def end(self, trainer):
        super().end(trainer)
        self._save_model(trainer.model, trainer.training_dir, "final")
        trainer.log("Model saved in %s." % os.path.split(trainer.training_dir)[1])


def LearningRateCallback(PeriodicCallback):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError

    def begin(self, trainer):
        super().begin(trainer)

    def after_train_step(self, trainer, step):
        super().after_train_step(trainer, step)

    def end(self, trainer):
        super().end(trainer)


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
        learning_rate = hparams.optimizer_lr
        if optimizer == optim.SGD:
            param_args = get_optimizer_parameters(
                ["momentum", "dampening", "weight_decay", "nesterov"])
        elif optimizer == optim.ASGD:
            param_args = get_optimizer_parameters(
                ["lambd", "alpha", "t0", "weight_decay"])
        elif optimizer == optim.RMSprop:
            param_args = get_optimizer_parameters(
                ["alpha", "eps", "weight_decay", "momentum", "centered"])
        elif optimizer == optim.Adagrad:
            param_args = get_optimizer_parameters(
                ["lr_decay", "weight_decay"])
        elif optimizer == optim.Adadelta:
            param_args = get_optimizer_parameters(
                ["rho", "eps", "weight_decay"])
        elif optimizer == optim.Adam:
            param_args = get_optimizer_parameters(
                ["betas", "eps", "weight_decay", "amsgrad"])
        elif optimizer == optim.SparseAdam:
            param_args = get_optimizer_parameters(
                ["betas", "eps"])
        elif optimizer == optim.Adamax:
            param_args = get_optimizer_parameters(
                ["betas", "eps", "weight_decay"])
        elif optimizer == optim.LBFGS:
            param_args = get_optimizer_parameters(
                ["max_iter", "max_eval", "tolerance_grad", "tolerance_change", 
                 "history_size", "line_search_fn"])
        elif optimizer == optim.Rprop:
            param_args = get_optimizer_parameters(
                ["etas", "step_sizes"]) 
        optimizer = optimizer_class(model.parameters(), lr=learning_rate,
                                    **param_args)
    else:
        optimizer = optimizer_class(**kwargs)

    return optimizer

