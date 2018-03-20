""""""

import statistics as stats

from .metrics import resolve_metrics


def infer_with_eval(model, hparams, batch_data, metrics):
    """
    Args:
        model: Module.
        hparams: Hparameters.
        batch_data: Tuple; a 2-tuple of the form (input_data, target_data), 
            where input_data is a data point of the form expected by model 
            on input and target_data is a Tensor.
        metrics: List; a sequence of callables which accept 
            a prediction/output tensor and a label/target tensor and return
            a single numeric value.
    Returns:
        A dictionary mapping metric to values. 
    """
    if isinstance(batch_data[1], list) or isinstance(batch_data[1], dict):
        raise ValueError("Unable to perform inference with evaluation on " 
            "complex target data---please use a custom function for this purpose.")
    else:
        target = batch_data[1]

    output = model(batch_data[0])

    return {metric: metric(output, target) for metric in metrics}


def evaluate(model, hparams, data_source, metrics, batch_size, *args, 
             eval_step_fn=infer_with_eval, steps=None, use_cuda=True, **kwargs):
    """
    Args:
        model: Module. 
        hparams: Hparameters.
        data_source: DataSource.
        metrics: List; a list of metric names or callables; each callable should 
            return exactly one numeric value. 
        batch_size: Int.
        args: Optional positional arguments passed to eval_step_fn.
        eval_step_fn: Callable; should accept model, hparams, batch_data 
            (concatenated, from data source), metrics, and any number of 
            additional arguments and keyword arguments and return 
            a dictionary mapping metrics to values.
        steps: Int; number of steps of evaluation; if None, then 
            the model is evaluated on the entire data source 
            (optional, default: None).
        use_cuda: Int; (optional, default: True).
        kwargs: Optional keyword arguments passed to eval_step_fn.
    Returns:
        A dictionary mapping the names of metrics to their mean value on 
        the evaluation data.
    """
    model.eval()
    metrics = resolve_metrics(metrics)
    step = 1
    results_per_step = []
    while True:
        batch_data, data_exhausted = data_source.get_next_batch(
            batch_size, use_cuda=use_cuda)
        results = eval_step_fn(model, hparams, batch_data, metrics, *args, **kwargs)
        results_per_step.append(results)
        step += 1
        if (steps is not None and step > steps) or data_exhausted:
            break

    eval_results = {}
    for metric in metrics:
        values = [results[metric] for results in results_per_step]
        eval_results[metric.__name__] = stats.mean(values)

    return eval_results