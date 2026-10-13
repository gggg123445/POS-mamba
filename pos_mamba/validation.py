"""Fail early on invalid data and patient leakage before model downloads."""
from pathlib import Path

import numpy as np
import pandas as pd

from .data import find_multiview_images


def validate_inputs(df, args):
    if df.empty:
        raise ValueError('CSV contains no records')
    id_col = next((c for c in ['record_id', 'uniq_ID'] if c in df), None)
    if id_col is None:
        raise ValueError('CSV requires record_id or uniq_ID for image matching')
    if df[id_col].isna().any() or df[id_col].astype(str).str.strip().eq('').any():
        raise ValueError(f'{id_col} contains missing/empty identifiers')
    if df[id_col].duplicated().any():
        raise ValueError(f'{id_col} must uniquely identify one vessel record')
    if not Path(args.image_root).is_dir():
        raise FileNotFoundError(f'Image directory does not exist: {args.image_root}')
    if args.group_col != 'auto':
        if (args.group_col not in df or df[args.group_col].isna().any()
                or df[args.group_col].astype(str).str.strip().eq('').any()):
            raise ValueError(f'Explicit group column is missing or contains nulls: {args.group_col}')
    for column in ['check_id', 'patient_id', 'PatientID', 'patient', 'PID']:
        if column in df and (df[column].isna().any() or df[column].astype(str).str.strip().eq('').any()):
            raise ValueError(f'{column} contains missing/empty patient identifiers')
    percentage = next((c for c in [args.stenosis_col, 'stenosis_percentage', 'stenosis_pct'] if c in df), None)
    if percentage:
        # Reject unknown/missing values rather than silently converting them to class 0.
        def valid_percentage(value):
            import re
            parts = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*%?\s*(?:-\s*(\d+(?:\.\d+)?)\s*%?)?\s*', str(value))
            return bool(parts) and all(0 <= float(p) <= 100 for p in parts.groups() if p is not None)
        if not df[percentage].apply(valid_percentage).all():
            raise ValueError(f'{percentage} must contain numbers/percentages/ranges within 0..100')
    else:
        from .utils import infer_label_col
        label_col = infer_label_col(df, args.label_col)
        if label_col is None:
            raise ValueError('CSV requires stenosis_percentage or a class label column')
        values = pd.to_numeric(df[label_col], errors='coerce')
        if not (np.isfinite(values).all() and values.between(0, 5).all() and values.eq(values.round()).all()):
            raise ValueError(f'{label_col} must contain integer classes 0..5')
    if args.use_hlgauss:
        if args.stenosis_col not in df:
            raise ValueError(f'HL-Gauss requires column: {args.stenosis_col}')
        if args.stenosis_col != percentage:
            vals = pd.to_numeric(df[args.stenosis_col], errors='coerce')
            if not (np.isfinite(vals).all() and vals.between(0, 100).all()):
                raise ValueError(f'{args.stenosis_col} must contain finite values within 0..100')
    missing = [str(rid) for rid in df[id_col].unique()
               if not find_multiview_images(args.image_root, str(rid), args.scan_type_filter)]
    if missing:
        raise FileNotFoundError(f'{len(missing)} records have no matching images; first IDs: {missing[:5]}')


def validate_split(train_df, val_df, test_df, group_col, batch_size):
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError('Train/validation/test splits must all contain records')
    if len(train_df) < batch_size:
        raise ValueError('Training split is smaller than batch_size; decrease --batch_size')
    groups = [set(frame[group_col].astype(str)) for frame in (train_df, val_df, test_df)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError('Patient groups overlap between train/validation/test splits')
