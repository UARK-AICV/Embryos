
def reset_metrics(metric_dict: dict):
    for metric in metric_dict:
        metric_dict[metric].reset()
