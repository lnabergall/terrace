""""""

import statistics as stats

from .metrics import resolve_metrics


def evaluate(model, hparams, data_source, metrics, batch_size, 
             *args, eval_step_fn=infer_with_eval, steps=None, **kwargs):
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
    if ((isinstance(target_data, list) and len(target_data) > 1) 
            or isinstance(target_data, dict)):
        raise ValueError("Unable to perform inference with evaluation on " 
            "complex target data---please use a custom function for this purpose.")
    elif isinstance(target_data, list):
        target = target_data[0]
    else:
        target = target_data

    output = model(input_data)
    
    return {metric(output, target) for metric in metrics}