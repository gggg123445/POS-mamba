"""Reproducibility, stenosis labels, patient grouping and CV summaries."""
import random
import re
import math
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

def json_safe(value):
    """Use null for undefined metrics so exports remain standard JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_stenosis_value(v):
    if pd.isna(v):
        return 0.0
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).strip().replace('%', '')
    if len(s) == 0:
        return 0.0
    if '-' in s:
        parts = [p.strip() for p in s.split('-') if p.strip()]
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except Exception:
                pass
        if len(nums) > 0:
            return float(sum(nums) / len(nums))
    try:
        return float(s)
    except Exception:
        return 0.0


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_score = None
        self.bad_epochs = 0

    def step(self, score):
        improved = False
        if self.best_score is None or score > (self.best_score + self.min_delta):
            self.best_score = float(score)
            self.bad_epochs = 0
            improved = True
        else:
            self.bad_epochs += 1
        should_stop = self.bad_epochs >= self.patience
        return improved, should_stop


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ['1', 'true', 'yes', 'y', 't']


def _extract_patient_id(record_id: str) -> str:
    if not isinstance(record_id, str):
        record_id = str(record_id)
    s = record_id.strip()
    m = re.match(r'^(\d+)', s)
    return m.group(1) if m else s


def ensure_patient_group_column(df: pd.DataFrame, group_col: str):
    df = df.copy()
    if group_col != 'auto':
        if group_col not in df.columns:
            raise ValueError(f'group_col not found: {group_col}')
        return df, group_col

    # Prefer provided patient identities to inference from vessel filenames.
    for c in ['patient_id', 'PatientID', 'patient', 'PID']:
        if c in df.columns:
            return df, c

    if 'check_id' in df.columns:
        df['patient_id'] = df['check_id'].astype(str)
        return df, 'patient_id'

    raise ValueError('An explicit patient identity column is required for patient-level splits')



def make_group_strat_table(df: pd.DataFrame, group_col: str, label_col='mapped_label'):
    if label_col not in df.columns:
        raise ValueError(f'Missing label column: {label_col}')
    g = df[[group_col, label_col]].copy()
    g[group_col] = g[group_col].astype(str)
    group_y = g.groupby(group_col)[label_col].agg(lambda x: int(pd.to_numeric(x, errors='coerce').fillna(0).max()))
    return pd.DataFrame({group_col: group_y.index.astype(str), 'y': group_y.values.astype(int)})


def infer_label_col(df: pd.DataFrame, preferred='mapped_label'):
    candidates = [
        preferred,
        'mapped_label',
        'multi_Class',
        'label',
        'class',
        'cad_rads',
        'cadrads',
        'CADRADS',
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def pct_to_class_6(pct: float) -> int:
    v = float(np.clip(pct, 0.0, 100.0))
    if v <= 0:
        return 0
    if v < 25:
        return 1
    if v < 50:
        return 2
    if v < 70:
        return 3
    if v < 100:
        return 4
    return 5


def ensure_mapped_label(df: pd.DataFrame, label_col: str, stenosis_col='stenosis_percentage'):
    df = df.copy()
    for pct_col in dict.fromkeys([stenosis_col, 'stenosis_percentage', 'stenosis_pct']):
        if pct_col in df.columns:
            y = df[pct_col].apply(parse_stenosis_value).apply(pct_to_class_6).astype(int)
            df['mapped_label'] = y
            return df, 'mapped_label', pct_col

    src = infer_label_col(df, preferred=label_col)
    if src is None:
        raise ValueError(
            f"Missing label column: {label_col}. Available columns do not include usable label fields "
            f"(e.g., stenosis_percentage/mapped_label/multi_Class)."
        )

    y = pd.to_numeric(df[src], errors='coerce').fillna(0).astype(int).clip(lower=0, upper=5)
    df['mapped_label'] = y
    return df, 'mapped_label', src


def split_train_val_groups(train_groups_df: pd.DataFrame, group_col: str, val_ratio: float, seed: int):
    if len(train_groups_df) <= 2 or val_ratio <= 0:
        return train_groups_df[group_col].tolist(), []

    y = train_groups_df['y'].values
    g = train_groups_df[group_col].values

    def _random_split(groups, ratio, rs):
        groups = np.array(groups)
        if len(groups) <= 1:
            return groups.tolist(), []
        rng = np.random.RandomState(rs)
        idx = np.arange(len(groups))
        rng.shuffle(idx)
        n_val = int(round(len(groups) * float(ratio)))
        n_val = min(max(1, n_val), len(groups) - 1)
        va_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        return groups[tr_idx].tolist(), groups[va_idx].tolist()

    n_classes = len(np.unique(y))
    min_val_groups = max(n_classes, int(round(len(train_groups_df) * val_ratio)))
    val_ratio_eff = min(0.5, max(float(min_val_groups) / max(len(train_groups_df), 1), val_ratio))

    if len(train_groups_df) * val_ratio_eff < n_classes:
        return g.tolist(), []

    class_counts = pd.Series(y).value_counts()
    if len(class_counts) == 0:
        return g.tolist(), []
    if int(class_counts.min()) < 2:
        return _random_split(g, val_ratio_eff, seed)

    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_eff, random_state=seed)
        tr_idx, va_idx = next(sss.split(g, y))
        return g[tr_idx].tolist(), g[va_idx].tolist()
    except ValueError:
        return _random_split(g, val_ratio_eff, seed)


def pick_metric(metrics, metric_key):
    if metric_key == 't1_f1_macro':
        return float(metrics['T1_F1'])
    if metric_key == 't1_mcc':
        return float(metrics['T1_MCC'])
    if metric_key == 't1_kappa':
        return float(metrics['T1_Kappa'])
    return float(metrics['T1_ACC'])


def select_monitor_score(metrics, val_loss, args):
    if args.save_best_on == 'composite':
        primary = pick_metric(metrics, args.composite_metric)
        return float(primary - args.composite_loss_weight * float(val_loss))
    return pick_metric(metrics, args.save_best_on)


def aggregate_cv_results(rows):
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c.startswith('test_') or c == 'best_score']
    agg = {}
    for c in metric_cols:
        vals = pd.to_numeric(df[c], errors='coerce').dropna().values
        if len(vals) == 0:
            continue
        agg[f'cv_mean_{c}'] = float(np.mean(vals))
        agg[f'cv_std_{c}'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return df, agg
