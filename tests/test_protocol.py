import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.metrics import f1_score
from torchvision.transforms import ToTensor

from examples.make_synthetic_data import generate
from pos_mamba import cli
from pos_mamba.data import VesselLongImageDatasetNativeNoPad
from pos_mamba.metrics import classification_metrics, patient_metrics


def test_binary_f1_is_macro_and_auc_uses_probability():
    y, pred = [0, 0, 0, 1], [0, 0, 1, 1]
    probs = [[.9, .1], [.8, .2], [.3, .7], [.1, .9]]
    result = classification_metrics(y, pred, probs, 2)
    assert result['F1'] == f1_score(y, pred, average='macro')
    assert result['AUC'] == 1 and result['SENS'] == 1 and result['PPV'] == .5
    absent = classification_metrics([0, 0], [0, 0], [[1, 0], [1, 0]], 2)
    assert np.isnan(absent['AUC']) and absent['SENS'] == 0


def test_patient_metrics_aggregate_only_available_vessels():
    frame = pd.DataFrame({'record_id': ['A_LAD', 'A_RCA', 'B_LAD'], 'patient_id': ['A', 'A', 'B']})
    results = {'record_ids': [['A_LAD', 'A_RCA', 'B_LAD']],
               'targets': [[1, 4, 0], [0, 1, 0]], 'preds': [[2, 3, 1], [0, 1, 0]]}
    scores, rows = patient_metrics(results, frame, 'patient_id')
    assert scores['PATIENT_COUNT'] == 2
    assert rows.set_index('patient_id').loc['A', 'T1_true'] == 4
    assert rows.set_index('patient_id').loc['A', 'T1_pred'] == 3


def test_sacr_transposes_then_concatenates_without_resizing(tmp_path):
    paths = []
    source = np.arange(60 * 44, dtype=np.uint16).reshape(60, 44).astype(np.uint8)
    for i in range(2):
        path = tmp_path / f'view{i}.png'
        Image.fromarray(source).save(path)
        paths.append(str(path))
    dataset = VesselLongImageDatasetNativeNoPad(pd.DataFrame(), str(tmp_path), ToTensor())
    image = dataset._build_long_image_tensor(paths)
    expected = source[:, 2:42].T / 255
    assert tuple(image.shape) == (1, 40, 120)
    np.testing.assert_allclose(image[0, :, :60], expected, atol=1e-7)


def test_explicit_development_manifests_are_replayed_and_reject_unknown_patients(tmp_path):
    path = tmp_path / 'development_groups.json'
    group_df = pd.DataFrame({'patient_id': ['A', 'B', 'C'], 'y': [0, 1, 2]})
    args = SimpleNamespace(development_splits_file=str(path), n_splits=2)
    splits = [{'train_groups': ['A', 'B'], 'val_groups': ['C']},
              {'train_groups': ['A', 'C'], 'val_groups': ['B']}]
    path.write_text(json.dumps({'splits': splits}))
    assert list(cli.development_partitions(group_df, args, 'patient_id')) == [({'A', 'B'}, {'C'}), ({'A', 'C'}, {'B'})]
    splits[0]['val_groups'] = ['HOLDOUT']
    path.write_text(json.dumps({'splits': splits}))
    with pytest.raises(ValueError, match='development cohort'):
        list(cli.development_partitions(group_df, args, 'patient_id'))


def test_complete_synthetic_run_selects_before_holdout_and_has_no_leakage(tmp_path, fake_backbone, monkeypatch):
    metadata, images = generate(tmp_path / 'data')
    output = tmp_path / 'output'
    original_evaluate = cli.evaluate_checkpoint
    evaluated = []
    def check_selection(*args):
        selection = json.loads((output / 'primary_selection.json').read_text())
        assert selection['selection_metric'] == 'validation_T1_macro_F1'
        assert all((output / f'fold_{fold}/best_model_fold{fold}.pth').exists() for fold in [1, 2, 3])
        evaluated.append(args[1])
        return original_evaluate(*args)
    monkeypatch.setattr(cli, 'evaluate_checkpoint', check_selection)
    argv = ['--csv', str(metadata), '--image_root', str(images), '--save_path', str(output),
            '--group_col', 'patient_id', '--epochs', '1', '--num_workers', '0', '--device', 'cpu',
            '--feature_proj_dim', '16', '--towers_hidden', '8', '--wavelet_freq_hidden', '8',
            '--hl_aux_hidden', '8', '--enable_aug', 'false']
    cli.main(argv)
    assert evaluated == [1, 2, 3]
    test_sets = []
    for fold in [1, 2, 3]:
        groups = json.loads((output / f'fold_{fold}/split_groups.json').read_text())
        train, val, test = (set(groups[key]) for key in ['train_groups', 'val_groups', 'test_groups'])
        assert not train & val and not train & test and not val & test
        test_sets.append(test)
        for task, count in [(1, 24), (2, 24), (3, 24), (4, 12), (5, 12), (6, 12)]:
            predictions = pd.read_csv(output / f'fold_{fold}/predictions_fold{fold}_T{task}.csv')
            assert len(predictions) == count
        assert len(pd.read_csv(output / f'fold_{fold}/patient_predictions.csv')) == 12
    assert test_sets[0] == test_sets[1] == test_sets[2]
    metrics = pd.read_csv(output / 'development_split_metrics.csv')
    summary = json.loads((output / 'summary.json').read_text())
    selected = metrics.loc[metrics.best_val_t1_f1_macro.idxmax()]
    assert summary['primary_split'] == selected['fold']
    assert summary['native_no_resize_pad']
    assert summary['cv_std_test_T1_F1'] == pytest.approx(metrics.test_T1_F1.std(ddof=1))
    with pytest.raises(FileExistsError):
        cli.main(argv)
