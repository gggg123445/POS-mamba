"""Train on development data; evaluate the holdout after primary selection."""
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score
from tqdm import tqdm

from .data import (VesselLongImageDatasetNativeNoPad, build_native_noresize_transforms,
                   collate_native_nopad, collate_native_nopad_with_stenosis)
from .models import POSMamba
from .losses import get_task_labels, compute_multitask_loss, hl_soft_ce_with_probs
from .metrics import classification_metrics, patient_metrics, TASK_CLASSES, BINARY_TASKS
from .utils import EarlyStopping, select_monitor_score

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    task_losses_sum = [0.0] * 6
    pbar = tqdm(loader, desc='Train', leave=False)
    for batch in pbar:
        if len(batch) == 4:
            images, labels, _stenosis, _rid = batch
        else:
            images, labels, _rid = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(images)
        if isinstance(preds, tuple):
            preds = preds[0]
        task_labels, masks = get_task_labels(labels)
        loss, losses = compute_multitask_loss(preds, task_labels, masks)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        for i, l in enumerate(losses):
            task_losses_sum[i] += l
        pbar.set_postfix({'loss': float(loss.item())})

    n = max(len(loader), 1)
    return total_loss / n, [l / n for l in task_losses_sum]


@torch.inference_mode()
def evaluate(model, loader, device, return_loss=False):
    model.eval()
    targets, predictions, probabilities, identifiers = [[[] for _ in range(6)] for _ in range(4)]
    total_loss, batches = 0.0, 0
    for batch in tqdm(loader, desc='Eval', leave=False):
        images, base_labels, rids = batch[0], batch[1], batch[-1]
        base_labels = base_labels.to(device)
        output = model(images.to(device))
        logits = output[0] if isinstance(output, tuple) else output
        labels, masks = get_task_labels(base_labels)
        if return_loss:
            loss, _ = compute_multitask_loss(logits, labels, masks)
            total_loss += float(loss)
            batches += 1
        for task in range(6):
            if task in BINARY_TASKS:
                positive = torch.sigmoid(logits[task].reshape(-1))
                probs = torch.stack([1 - positive, positive], dim=1)
                pred = (positive > 0.5).long()
            else:
                probs = torch.softmax(logits[task], dim=1)
                pred = probs.argmax(dim=1)
            mask = masks[task] if masks[task] is not None else torch.ones_like(base_labels, dtype=torch.bool)
            targets[task].extend(labels[task].reshape(-1)[mask].long().cpu().tolist())
            predictions[task].extend(pred[mask].cpu().tolist())
            probabilities[task].extend(probs[mask].cpu().tolist())
            identifiers[task].extend([rids[i] for i in mask.cpu().nonzero().reshape(-1).tolist()])
    results = {'targets': targets, 'preds': predictions,
               'probabilities': probabilities, 'record_ids': identifiers}
    for task, classes in enumerate(TASK_CLASSES):
        scores = classification_metrics(targets[task], predictions[task], probabilities[task], classes)
        results.update({f'T{task+1}_{key}': value for key, value in scores.items()})
    results['T1_Kappa'] = float(cohen_kappa_score(
        targets[0], predictions[0], labels=list(range(6)), weights='linear'))
    if return_loss:
        results['LOSS'] = total_loss / max(batches, 1)
    return results


@torch.inference_mode()
def evaluate_hlgauss(model, loader, device):
    model.eval()
    all_true = []
    all_pred = []

    for images, _labels, stenosis, _rid in tqdm(loader, desc='Eval-HL', leave=False):
        images = images.to(device)
        stenosis = stenosis.to(device)
        _preds, hl_logits = model(images)
        probs = torch.softmax(hl_logits, dim=1)
        pred_st = model.hl_gauss_from_probs(probs)
        all_true.append(stenosis.detach().cpu())
        all_pred.append(pred_st.detach().cpu())

    if len(all_true) == 0:
        return {'HL_MAE': 0.0, 'HL_RMSE': 0.0, 'HL_CADRADS_ACC': 0.0}

    y_true = torch.cat(all_true, dim=0).numpy()
    y_pred = torch.cat(all_pred, dim=0).numpy()
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    y_true_cls = [POSMamba.stenosis_to_cadrads_label(float(x)) for x in y_true]
    y_pred_cls = [POSMamba.stenosis_to_cadrads_label(float(x)) for x in y_pred]
    cls_acc = float(accuracy_score(y_true_cls, y_pred_cls)) if len(y_true_cls) > 0 else 0.0
    return {'HL_MAE': mae, 'HL_RMSE': rmse, 'HL_CADRADS_ACC': cls_acc}


def train_one_epoch_with_hlgauss(model, loader, optimizer, device, hl_sigma=10.0, hl_weight=0.9):
    model.train()
    total_loss = 0.0
    total_main = 0.0
    total_hl = 0.0
    task_losses_sum = [0.0] * 6

    pbar = tqdm(loader, desc='Train', leave=False)
    for images, labels, stenosis, _rid in pbar:
        images = images.to(device)
        labels = labels.to(device)
        stenosis = stenosis.to(device)

        optimizer.zero_grad()
        preds, hl_logits = model(images)
        task_labels, masks = get_task_labels(labels)
        main_loss, losses = compute_multitask_loss(preds, task_labels, masks)

        hl_targets = model.hl_gauss_to_probs(stenosis, sigma=hl_sigma)
        hl_loss = hl_soft_ce_with_probs(hl_logits, hl_targets)

        loss = (1 - float(hl_weight)) * main_loss + float(hl_weight) * hl_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_main += float(main_loss.item())
        total_hl += float(hl_loss.item())
        for i, l in enumerate(losses):
            task_losses_sum[i] += l

    n = max(len(loader), 1)
    avg_task_losses = [l / n for l in task_losses_sum]
    return total_loss / n, total_main / n, total_hl / n, avg_task_losses


def make_model(args, device):
    return POSMamba(
        model_name=args.model_name,
        feature_proj_dim=args.feature_proj_dim,
        towers_hidden=args.towers_hidden,
        use_hlgauss=args.use_hlgauss,
        hl_bins_mode=args.hl_bins_mode,
        hl_aux_hidden=args.hl_aux_hidden,
        hl_aux_dropout=args.hl_aux_dropout,
        repeat_gray_to_3ch=args.repeat_gray_to_3ch,
        use_wavelet_freq_branch=args.use_wavelet_freq_branch,
        wavelet_use_hh=args.wavelet_use_hh,
        wavelet_freq_hidden=args.wavelet_freq_hidden,
        wavelet_fusion_mode=args.wavelet_fusion_mode,
        wavelet_fusion_alpha=args.wavelet_fusion_alpha,
        wavelet_levels=args.wavelet_levels,
        wavelet_use_ll=args.wavelet_use_ll,
        wavelet_band_norm=args.wavelet_band_norm,
        wavelet_band_attention=args.wavelet_band_attention,
        wavelet_band_attn_temperature=args.wavelet_band_attn_temperature,
        model_revision=args.model_revision,
    ).to(device)


def make_dataset(args, frame, transform):
    return VesselLongImageDatasetNativeNoPad(
        frame, args.image_root, transform,
        stenosis_col=args.stenosis_col if args.use_hlgauss else None,
        num_views=args.num_views, sampling_method=args.sampling_method,
        scan_type_filter=args.scan_type_filter, center_crop_h_ratio=args.center_crop_h_ratio,
        center_crop_w_px=args.center_crop_w_px, seq_fixed_h=args.seq_fixed_h,
        seq_post_interp_h=args.seq_post_interp_h, min_side_fallback=args.min_side_fallback)


def make_loader(args, dataset, training=False):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=training,
                      num_workers=args.num_workers, drop_last=training,
                      collate_fn=collate_native_nopad_with_stenosis if args.use_hlgauss else collate_native_nopad)


def run_one_fold(args, fold_id, train_df, val_df, test_df, device):
    fold_dir = os.path.join(args.save_path, f'fold_{fold_id}')
    best_model_path = os.path.join(fold_dir, f'best_model_fold{fold_id}.pth')
    if os.path.exists(best_model_path):
        raise FileExistsError(f'Checkpoint exists: {best_model_path}. Choose a new --save_path.')
    os.makedirs(fold_dir, exist_ok=True)

    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(fold_dir, f'run_fold{fold_id}_{run_stamp}.txt')
    with open(log_path, 'w') as f:
        f.write(json.dumps(vars(args), indent=2, ensure_ascii=True) + '\n')
        f.write(f'fold={fold_id}\ntrain={len(train_df)} val={len(val_df)} test={len(test_df)}\n\n')

    train_transform, val_transform = build_native_noresize_transforms(enable_aug=bool(args.enable_aug))

    train_loader = make_loader(args, make_dataset(args, train_df, train_transform), training=True)
    val_loader = make_loader(args, make_dataset(args, val_df, val_transform))
    if len(train_loader) == 0:
        raise ValueError('Training split is smaller than batch_size')

    model = make_model(args, device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    best_score = -999.0
    best_val_t1_f1 = -999.0
    best_epoch = 0
    best_model_path = os.path.join(fold_dir, f'best_model_fold{fold_id}.pth')
    hist = []

    for epoch in range(args.epochs):
        if args.use_hlgauss:
            train_loss, train_main_loss, train_hl_loss, train_task_losses = train_one_epoch_with_hlgauss(
                model, train_loader, optimizer, device,
                hl_sigma=args.hl_sigma, hl_weight=args.hl_loss_weight,
            )
        else:
            train_loss, train_task_losses = train_one_epoch(model, train_loader, optimizer, device)
            train_main_loss, train_hl_loss = train_loss, 0.0

        scheduler.step()

        val_res = evaluate(model, val_loader, device, return_loss=True)
        val_loss = float(val_res.pop('LOSS', 0.0))
        val_score = select_monitor_score(val_res, val_loss, args)
        hl_res = evaluate_hlgauss(model, val_loader, device) if args.use_hlgauss else {'HL_MAE': 0.0, 'HL_RMSE': 0.0, 'HL_CADRADS_ACC': 0.0}

        row = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_main_loss': train_main_loss,
            'train_hl_loss': train_hl_loss,
            'val_loss': val_loss,
            'monitor_score': val_score,
            'val_HL_MAE': hl_res['HL_MAE'],
            'val_HL_RMSE': hl_res['HL_RMSE'],
            'val_HL_CADRADS_ACC': hl_res['HL_CADRADS_ACC'],
        }
        for i, l in enumerate(train_task_losses):
            row[f'train_task{i+1}_loss'] = l
        for k, v in val_res.items():
            if not isinstance(v, list):
                row[f'val_{k}'] = v
        hist.append(row)

        print(f'[Fold {fold_id}] Epoch {epoch+1}/{args.epochs} | train={train_loss:.4f} val={val_loss:.4f} score={val_score:.4f}')

        improved, should_stop = early_stopper.step(val_score)
        if improved:
            best_score = val_score
            best_val_t1_f1 = val_res['T1_F1']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
        if should_stop:
            break

    pd.DataFrame(hist).to_csv(os.path.join(fold_dir, f'train_history_fold{fold_id}.csv'), index=False)

    return {'fold': fold_id, 'best_score': best_score,
            'best_val_t1_f1_macro': best_val_t1_f1, 'best_epoch': best_epoch}


def evaluate_checkpoint(args, fold_id, test_df, device, group_col):
    fold_dir = os.path.join(args.save_path, f'fold_{fold_id}')
    _, transform = build_native_noresize_transforms(enable_aug=False)
    loader = make_loader(args, make_dataset(args, test_df, transform))
    model = make_model(args, device)
    model.load_state_dict(torch.load(
        os.path.join(fold_dir, f'best_model_fold{fold_id}.pth'), map_location='cpu', weights_only=True))
    result = evaluate(model, loader, device)
    patient_scores, patient_rows = patient_metrics(result, test_df, group_col)
    output = {f'test_{key}': value for key, value in result.items() if not isinstance(value, list)}
    output.update({f'test_{key}': value for key, value in patient_scores.items()})
    if args.use_hlgauss:
        output.update({f'test_{key}': value for key, value in evaluate_hlgauss(model, loader, device).items()})
    pd.DataFrame([output]).to_csv(os.path.join(fold_dir, f'test_metrics_fold{fold_id}.csv'), index=False)
    patient_rows.to_csv(os.path.join(fold_dir, 'patient_predictions.csv'), index=False)
    id_col = 'record_id' if 'record_id' in test_df else 'uniq_ID'
    patients = test_df.set_index(id_col)[group_col].astype(str).to_dict()
    for task, classes in enumerate(TASK_CLASSES):
        rows = []
        for rid, y, pred, probs in zip(result['record_ids'][task], result['targets'][task],
                                      result['preds'][task], result['probabilities'][task]):
            row = {'record_id': rid, 'patient_id': patients[rid], 'true_class': y,
                   'pred_class': pred, 'reference_class_probability': probs[y]}
            row.update({f'prob_cls{c}': probs[c] for c in range(classes)})
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(fold_dir, f'predictions_fold{fold_id}_T{task+1}.csv'), index=False)
    return output
