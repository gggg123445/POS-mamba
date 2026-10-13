"""Manuscript metrics: macro F1, probability AUC and available-vessel aggregation."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, matthews_corrcoef, roc_auc_score)

TASK_CLASSES = [6, 2, 4, 2, 2, 3]
BINARY_TASKS = {1, 3, 4}  # zero-based task indices


def classification_metrics(targets, predictions, probabilities, n_classes):
    y = np.asarray(targets, dtype=int)
    pred = np.asarray(predictions, dtype=int)
    labels = list(range(n_classes))
    if not len(y):
        return {key: float('nan') for key in ['ACC', 'F1', 'MCC', 'AUC', 'SENS', 'PPV', 'NPV']}
    cm = confusion_matrix(y, pred, labels=labels)
    tp = cm.diagonal().astype(float)
    fp, fn = cm.sum(axis=0) - tp, cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b > 0)
    sens, ppv, npv = divide(tp, tp + fn), divide(tp, tp + fp), divide(tn, tn + fn)
    # Binary operating characteristics refer to class 1; multiclass uses macro OVR.
    select = (lambda a: float(a[1])) if n_classes == 2 else (lambda a: float(a.mean()))
    probs = np.asarray(probabilities)
    auc = float('nan')
    # Missing reference classes make the full macro OVR AUC undefined.
    if len(np.unique(y)) == n_classes:
        if n_classes == 2:
            auc = float(roc_auc_score(y, probs[:, 1]))
        else:
            auc = float(np.mean([roc_auc_score((y == c).astype(int), probs[:, c])
                                 for c in labels]))
    return {
        'ACC': float(accuracy_score(y, pred)),
        'F1': float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
        'MCC': (0.0 if len(np.unique(np.concatenate([y, pred]))) == 1
                else float(matthews_corrcoef(y, pred))), 'AUC': auc,
        'SENS': select(sens), 'PPV': select(ppv), 'NPV': select(npv),
    }


def patient_metrics(results, frame, group_col):
    """Maximum true/predicted grade among available vessels, with no synthetic vessels."""
    id_col = 'record_id' if 'record_id' in frame else 'uniq_ID'
    lookup = frame.set_index(id_col)[group_col].astype(str).to_dict()
    rows = pd.DataFrame({
        'patient_id': [lookup[rid] for rid in results['record_ids'][0]],
        'T1_true': results['targets'][0], 'T1_pred': results['preds'][0],
        'T2_true': results['targets'][1], 'T2_pred': results['preds'][1],
    })
    grouped = rows.groupby('patient_id', sort=True).max()
    metrics = {'PATIENT_COUNT': len(grouped)}
    for task, classes in [(1, 6), (2, 2)]:
        y, pred = grouped[f'T{task}_true'], grouped[f'T{task}_pred']
        metrics[f'PATIENT_T{task}_ACC'] = float(accuracy_score(y, pred))
        metrics[f'PATIENT_T{task}_F1'] = float(f1_score(
            y, pred, labels=list(range(classes)), average='macro', zero_division=0))
    metrics['PATIENT_T1_Kappa'] = float(cohen_kappa_score(
        grouped['T1_true'], grouped['T1_pred'], labels=list(range(6)), weights='linear'))
    return metrics, grouped.reset_index()
