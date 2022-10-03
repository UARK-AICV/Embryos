import pandas as pd 
from metrics.evaluator import get_precision_recall_by_class, get_accuracy_per_class
from sklearn.metrics import precision_score, recall_score, f1_score

def print_metrics_segment(segment_metrics, n_classes=4, logger = None):
    print_str = ''
    df_score = {
        'Class': list(range(n_classes)),
        'Rec': [], 'Prec': []
    }
    for c in range(n_classes):
        df_score['Rec'].append(f"{segment_metrics['rec'][c].avg:.4f}")
        df_score['Prec'].append(f"{segment_metrics['prec'][c].avg:.4f}")
    df_score = pd.DataFrame.from_dict(df_score)
    if logger is None:
        print(df_score)
    else:
        logger.info(df_score)
    
    return df_score


def print_metrics_frames(metric_dict, n_classes=4, logger = None, summary=False):
    print_func = logger.info if logger is not None else print
    print_str = ''

    for metric_name, metric_score in metric_dict.items():
        if summary:
            print_func(f'------------- [{metric_name}] -------------')
        for metric in metric_score:
            if metric  == 'confusion_matrix':
                if summary:
                    metric_score[metric].summary(logger=logger)
                continue 
            if summary:
                print_func(f'=== Global {metric} ===')
                metric_score[metric].summary(logger=logger)
            else:
                print_str += (f'[{metric_name}] {metric}: {metric_score[metric].value():.4f}, ')
        print_str += '\n'

    if not summary:
        print_func(print_str)
        
    df_score = {
        'Class': list(range(n_classes))
    }
    
    for metric_name, metric_score in metric_dict.items():
        df_score[f'{metric_name}-Rec'] = []
        df_score[f'{metric_name}-Prec'] = []
        df_score[f'{metric_name}-Acc'] = []

        prec_mat, rec_mat = get_precision_recall_by_class(metric_score['confusion_matrix'].value())
        acc_class = get_accuracy_per_class(metric_score['confusion_matrix'].value())
        for c in range(n_classes):
            df_score[f'{metric_name}-Acc'].append(f'{acc_class[c]:.4f}')
            df_score[f'{metric_name}-Rec'].append(f'{rec_mat[c][c]:.4f}')
            df_score[f'{metric_name}-Prec'].append(f'{prec_mat[c][c]:.4f}')
    
    print_func(f'=== Per-class performance ===')
    df_score = pd.DataFrame.from_dict(df_score)
    print_func(df_score)
    
    return df_score

def print_metrics(frame_metrics, width_metrics, n_classes=4):
    print_str = ''
    for metric in frame_metrics:
        if metric  == 'confusion_matrix':
            continue
        print_str += (f'[Frame] {metric}: {frame_metrics[metric].value():.4f}, ')
    print_str += '\n'

    for metric in frame_metrics:
        if metric  == 'confusion_matrix':
            continue
        print_str += (f'[Width] {metric}: {width_metrics[metric].value():.4f}, ')
    print_str += '\n'
    
    frame_prec_mat, frame_rec_mat = get_precision_recall_by_class(frame_metrics['confusion_matrix'].value())
    width_prec_mat, width_rec_mat = get_precision_recall_by_class(width_metrics['confusion_matrix'].value())
    
    for c in range(n_classes):
        print_str += (
            f"\nClass {c}, F-prec: {frame_prec_mat[c][c]:.4f}, W-prec: {width_prec_mat[c][c]:.4f}, " + \
                f" F-rec: {frame_rec_mat[c][c]:.4f}, W-rec: {width_rec_mat[c][c]:.4f}"
        )
    print(print_str)
