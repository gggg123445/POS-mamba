"""Command-line entry point for patient-grouped cross validation."""
import os
import json
import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .engine import run_one_fold, evaluate_checkpoint
from .validation import validate_inputs, validate_split
from .utils import (str2bool, set_seed, ensure_mapped_label, ensure_patient_group_column,
                    make_group_strat_table, split_train_val_groups, aggregate_cv_results, json_safe)

def development_partitions(group_df, args, group_col):
    all_groups = set(group_df[group_col].astype(str))
    if args.development_splits_file:
        with open(args.development_splits_file, encoding='utf-8') as stream:
            payload = json.load(stream)
        splits = payload['splits']
        if len(splits) != args.n_splits:
            raise ValueError('Development manifest split count differs from n_splits')
        for split in splits:
            train = set(map(str, split['train_groups']))
            val = set(map(str, split['val_groups']))
            if not train or not val or train & val or train | val != all_groups:
                raise ValueError('Each manifest split must partition the entire development cohort without overlap')
            yield train, val
        return
    splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(group_df[group_col], group_df['y']), start=1):
        train, extra_val = split_train_val_groups(
            group_df.iloc[train_idx].reset_index(drop=True), group_col, args.val_ratio, args.seed + fold)
        val = set(extra_val) | set(group_df.iloc[val_idx][group_col])
        yield set(train) - val, val


def main(argv=None):
    parser = argparse.ArgumentParser(description='POS-Mamba: three development splits on one fixed patient holdout')
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='./outputs/pos_mamba_v2')

    parser.add_argument('--model_name', type=str, default='nvidia/MambaVision-T-1K')
    parser.add_argument('--model_revision', type=str, default=None,
                        help='Optional Hugging Face commit/tag for the model and remote code')
    parser.add_argument('--feature_proj_dim', type=int, default=512)
    parser.add_argument('--towers_hidden', type=int, default=128)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)

    parser.add_argument('--num_views', type=int, default=30)
    parser.add_argument('--sampling_method', type=str, default='uniform', choices=['uniform', 'random', 'first', 'step'])
    parser.add_argument('--scan_type_filter', type=str, default=None)

    parser.add_argument('--center_crop_h_ratio', type=float, default=1.0)
    parser.add_argument('--center_crop_w_px', type=int, default=40)
    parser.add_argument('--seq_fixed_h', type=int, default=0)
    parser.add_argument('--seq_post_interp_h', type=int, default=0)
    parser.add_argument('--min_side_fallback', type=int, default=32)
    parser.add_argument('--enable_aug', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--repeat_gray_to_3ch', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--use_wavelet_freq_branch', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--wavelet_use_hh', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--wavelet_use_ll', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--wavelet_levels', type=int, default=4)
    parser.add_argument('--wavelet_band_norm', type=str, default='false', choices=['true', 'false'])
    parser.add_argument('--wavelet_band_attention', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--wavelet_band_attn_temperature', type=float, default=1.0)
    parser.add_argument('--wavelet_freq_hidden', type=int, default=256)
    parser.add_argument('--wavelet_fusion_mode', type=str, default='gated', choices=['concat', 'sum', 'gated'])
    parser.add_argument('--wavelet_fusion_alpha', type=float, default=0.2)

    parser.add_argument('--gpu', type=int, default=0, help='Logical CUDA device index')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--n_splits', type=int, default=3)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--test_groups_file', type=str, default='')
    parser.add_argument('--development_splits_file', type=str, default='',
                        help='Optional exact private train/validation patient manifests')
    parser.add_argument('--save_test_groups_file', type=str, default='')
    parser.add_argument('--val_ratio', type=float, default=0.0)
    parser.add_argument('--group_col', type=str, default='auto')
    parser.add_argument('--label_col', type=str, default='mapped_label')

    parser.add_argument('--save_best_on', type=str, default='t1_f1_macro', choices=['t1_acc', 't1_f1_macro', 't1_mcc', 't1_kappa', 'composite'])
    parser.add_argument('--composite_metric', type=str, default='t1_f1_macro', choices=['t1_acc', 't1_f1_macro', 't1_mcc', 't1_kappa'])
    parser.add_argument('--composite_loss_weight', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--min_delta', type=float, default=0.0)

    parser.add_argument('--use_hlgauss', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--stenosis_col', type=str, default='stenosis_percentage')
    parser.add_argument('--hl_bins_mode', type=str, default='cadrads', choices=['cadrads', 'uniform101'])
    parser.add_argument('--hl_sigma', type=float, default=10.0)
    parser.add_argument('--hl_loss_weight', type=float, default=0.9)
    parser.add_argument('--hl_aux_hidden', type=int, default=256)
    parser.add_argument('--hl_aux_dropout', type=float, default=0.5)

    args = parser.parse_args(argv)
    args.use_hlgauss = str2bool(args.use_hlgauss)
    args.enable_aug = str2bool(args.enable_aug)
    args.repeat_gray_to_3ch = str2bool(args.repeat_gray_to_3ch)
    args.use_wavelet_freq_branch = str2bool(args.use_wavelet_freq_branch)
    args.wavelet_use_hh = str2bool(args.wavelet_use_hh)
    args.wavelet_use_ll = str2bool(args.wavelet_use_ll)
    args.wavelet_band_norm = str2bool(args.wavelet_band_norm)
    args.wavelet_band_attention = str2bool(args.wavelet_band_attention)

    if args.epochs < 1 or args.batch_size < 1 or args.n_splits < 2 or args.num_workers < 0:
        parser.error('epochs/batch_size must be positive; n_splits >= 2; num_workers >= 0')
    if not 0 < args.test_ratio < 1 or not 0 <= args.val_ratio < 1:
        parser.error('test_ratio must be in (0,1); val_ratio must be in [0,1)')
    if args.wavelet_levels < 1 or args.hl_sigma <= 0 or not 0 <= args.hl_loss_weight <= 1:
        parser.error('wavelet_levels/hl_sigma must be positive; hl_loss_weight must be in [0,1]')
    if args.wavelet_band_attn_temperature <= 0:
        parser.error('wavelet_band_attn_temperature must be positive')
    if not 0 < args.center_crop_h_ratio <= 1 or not 1 <= args.num_views <= 30:
        parser.error('center_crop_h_ratio must be in (0,1]; num_views must be in 1..30')
    if args.lr <= 0 or args.weight_decay < 0 or args.patience < 1:
        parser.error('lr/patience must be positive; weight_decay must be nonnegative')
    if not args.repeat_gray_to_3ch:
        parser.error('The pretrained MambaVision backbone requires three-channel input')
    if args.test_groups_file and not Path(args.test_groups_file).is_file():
        parser.error(f'test_groups_file does not exist: {args.test_groups_file}')
    if args.development_splits_file and not Path(args.development_splits_file).is_file():
        parser.error(f'development_splits_file does not exist: {args.development_splits_file}')
    output_dir = Path(args.save_path)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f'Output path is not empty: {output_dir}. Choose a new --save_path.')

    set_seed(args.seed)
    use_cuda = args.device != 'cpu' and torch.cuda.is_available()
    if args.device == 'cuda' and not use_cuda:
        parser.error('CUDA was requested but is unavailable')
    if use_cuda and not 0 <= args.gpu < torch.cuda.device_count():
        parser.error('gpu must identify an available logical CUDA device')
    device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')

    df = pd.read_csv(args.csv, dtype={c: str for c in
                     ['record_id', 'uniq_ID', 'check_id', 'patient_id', 'PatientID', 'patient', 'PID', args.group_col]})
    validate_inputs(df, args)
    df, args.label_col, src_label_col = ensure_mapped_label(df, args.label_col, args.stenosis_col)
    if src_label_col != 'mapped_label':
        print(f"[Info] label column '{args.label_col}' prepared from source '{src_label_col}'.")
    df, group_col_name = ensure_patient_group_column(df, args.group_col)

    if args.use_hlgauss and args.stenosis_col not in df.columns:
        raise ValueError(f'CSV missing column: {args.stenosis_col}')

    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, 'run_args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=True)

    group_df = make_group_strat_table(df, group_col_name, args.label_col)
    groups = group_df[group_col_name].values
    y = group_df['y'].values

    if args.test_groups_file and os.path.exists(args.test_groups_file):
        with open(args.test_groups_file, 'r') as f:
            payload = json.load(f)
        test_groups_loaded = payload.get('test_groups', []) if isinstance(payload, dict) else payload
        holdout_test_groups = set(str(x) for x in test_groups_loaded)
        all_groups = set(group_df[group_col_name].astype(str).tolist())
        unknown = holdout_test_groups - all_groups
        if unknown:
            raise ValueError(f'Holdout file contains {len(unknown)} unknown patient groups')
        if len(holdout_test_groups) == 0:
            raise ValueError(f'No valid holdout test groups found in file: {args.test_groups_file}')
        trainval_group_df = group_df[~group_df[group_col_name].astype(str).isin(holdout_test_groups)].reset_index(drop=True)
    else:
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=args.test_ratio, random_state=args.seed)
        trainval_idx, test_idx = next(sss_test.split(groups, y))
        trainval_group_df = group_df.iloc[trainval_idx].reset_index(drop=True)
        holdout_test_groups = set(groups[test_idx].tolist())

    save_groups_path = args.save_test_groups_file.strip() if isinstance(args.save_test_groups_file, str) else ''
    if save_groups_path:
        os.makedirs(os.path.dirname(save_groups_path) or '.', exist_ok=True)
        payload = {
            'group_col': group_col_name,
            'test_ratio': float(args.test_ratio),
            'seed': int(args.seed),
            'test_groups': sorted([str(x) for x in holdout_test_groups]),
        }
        with open(save_groups_path, 'w') as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)

    fold_rows = []
    partitions = list(development_partitions(trainval_group_df, args, group_col_name))
    for fold, (tr_set, va_set) in enumerate(partitions, start=1):

        train_df = df[df[group_col_name].astype(str).isin(tr_set)].reset_index(drop=True)
        val_df = df[df[group_col_name].astype(str).isin(va_set)].reset_index(drop=True)
        test_df = df[df[group_col_name].astype(str).isin(holdout_test_groups)].reset_index(drop=True)

        validate_split(train_df, val_df, test_df, group_col_name, args.batch_size)
        split_dir = Path(args.save_path) / f'fold_{fold}'
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / 'split_groups.json').open('w', encoding='utf-8') as f:
            json.dump({'group_col': group_col_name, 'train_groups': sorted(tr_set),
                       'val_groups': sorted(va_set), 'test_groups': sorted(holdout_test_groups)}, f, indent=2)

        print(f'===== Fold {fold}/{args.n_splits} =====')
        print(f'train={len(train_df)} val={len(val_df)} test={len(test_df)}')
        row = run_one_fold(args, fold, train_df, val_df, test_df, device)
        fold_rows.append(row)

    # Freeze the primary checkpoint using development validation only, before holdout evaluation.
    primary = max(fold_rows, key=lambda row: row['best_val_t1_f1_macro'])
    selection = {'primary_split': primary['fold'], 'selection_metric': 'validation_T1_macro_F1',
                 'validation_score': primary['best_val_t1_f1_macro'],
                 'checkpoint': f"fold_{primary['fold']}/best_model_fold{primary['fold']}.pth"}
    with (Path(args.save_path) / 'primary_selection.json').open('w', encoding='utf-8') as f:
        json.dump(selection, f, indent=2)
    for row in fold_rows:
        row.update(evaluate_checkpoint(args, row['fold'], test_df, device, group_col_name))
    with (Path(args.save_path) / 'primary_metrics.json').open('w', encoding='utf-8') as f:
        json.dump(json_safe({**selection, **{k: v for k, v in primary.items() if k.startswith('test_')}}),
                  f, indent=2, allow_nan=False)
    fold_df, agg = aggregate_cv_results(fold_rows)
    fold_df.to_csv(os.path.join(args.save_path, 'development_split_metrics.csv'), index=False)

    summary = {
        'protocol': 'development_splits_same_fixed_holdout',
        **selection,
        'n_splits': args.n_splits,
        'test_ratio': args.test_ratio,
        'group_col': group_col_name,
        'model_name': args.model_name,
        'model_revision': args.model_revision,
        'repeat_gray_to_3ch': bool(args.repeat_gray_to_3ch),
        'use_wavelet_freq_branch': bool(args.use_wavelet_freq_branch),
        'wavelet_use_hh': bool(args.wavelet_use_hh),
        'wavelet_use_ll': bool(args.wavelet_use_ll),
        'wavelet_levels': int(args.wavelet_levels),
        'wavelet_band_norm': bool(args.wavelet_band_norm),
        'wavelet_band_attention': bool(args.wavelet_band_attention),
        'wavelet_band_attn_temperature': float(args.wavelet_band_attn_temperature),
        'wavelet_freq_hidden': int(args.wavelet_freq_hidden),
        'wavelet_fusion_mode': str(args.wavelet_fusion_mode),
        'wavelet_fusion_alpha': float(args.wavelet_fusion_alpha),
        'center_crop_h_ratio': args.center_crop_h_ratio,
        'center_crop_w_px': args.center_crop_w_px,
        'seq_fixed_h': args.seq_fixed_h,
        'seq_post_interp_h': args.seq_post_interp_h,
        'native_no_resize_pad': args.seq_fixed_h == 0 and args.seq_post_interp_h == 0,
        'long_image_direct_to_mambavision': True,
        'long_image_wavelet_dual_branch': True,
        'use_hlgauss': bool(args.use_hlgauss),
        **agg,
    }
    with open(os.path.join(args.save_path, 'summary.json'), 'w') as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=True, allow_nan=False)

    print('\n===== Fixed-holdout development split summary =====')
    for k, v in summary.items():
        if isinstance(v, float):
            print(f'{k}: {v:.6f}')
        else:
            print(f'{k}: {v}')


if __name__ == '__main__':
    main()
