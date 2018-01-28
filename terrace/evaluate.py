""""""

import statistics as stats

from .metrics import resolve_metrics


def evaluate(model, hparams, data_source, metrics, batch_size, 
             *args, eval_step_fn=infer_with_eval, steps=None, **kwargs):
    """
    Args:
        model: Module. 
        hparams: Hparameters.
        data_source: DataSource.
        metrics: List; a list of metric names or callables; each callable should 
            return exactly one numeric value. 
        batch_size: Int.
        args: Optional positional arguments passed to eval_step_fn.
        eval_step_fn: Callable; should accept model, hparams, input_data 
            (from data source), target_data (from data source), metrics,
            and any number of additional arguments and keyword arguments
            and return a dictionary mapping metrics to values. 
        steps: Int; number of steps of evaluation; if None, then 
            the model is evaluated on the entire data source 
            (optional, default: None).
        kwargs: Optional keyword arguments passed to eval_step_fn.
    Returns:
        A dictionary mapping the names of metrics to their mean value on 
        the evaluation data.
    """
    metrics = resolve_metrics(metrics)
    step = 1
    results_per_step = []
    while True:
        (input_data, target_data), data_exhausted = data_source.get_next_batch(
            batch_size, concat_batchwise=True)
        results = eval_step_fn(model, hparams, input_data, target_data, 
                               metrics, *args, **kwargs)
        results_per_step.append(results)
        step += 1
        if step > steps or data_exhausted:
            break

    eval_results = {}
    for metric in metrics:
        values = [results[metric] for results in results_per_step]
        eval_results[metric.__name__] = stats.mean(values)

    return eval_results


def infer_with_eval(model, hparams, input_data, target_data, metrics):
    """
    Args:
        model: Module.
        hparams: Hparameters.
        input_data: Should be a data point of the form expected by model on input.
        target_data: List or Tensor; if a list, should be a list containing 
            a single tensor. 
        metrics: List; a sequence of callables which accept 
            a prediction/output tensor and a label/target tensor and return
            a single numeric value.
    Returns:
        A dictionary mapping metric to values. 
    """
    if ((isinstance(target_data, list) and len(target_data) > 1) 
            or isinstance(target_data, dict)):
        raise ValueError("Unable to perform inference with evaluation on " 
            "complex target data---please use a custom function for this purpose.")
    elif isinstance(target_data, list):
        target = target_data[0]
    else:
        target = target_data

    output = model(input_data)

    return {metric: metric(output, target) for metric in metrics}