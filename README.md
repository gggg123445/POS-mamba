# POS-Mamba

POS-Mamba grades coronary diameter stenosis from ordered, multi-angle CPR projections. It combines a **Spatial-Angular Continuous Representation (SACR)**, MambaVision spatial semantics, a Haar frequency branch, a pixel-wise semantic alignment gate, six clinical-prior task heads, and probabilistic ordinal supervision.

[中文说明](docs/README_zh.md) · [Paper alignment and changes](docs/PAPER_ALIGNMENT.md) · [Third-party licenses](THIRD_PARTY_NOTICES.md)

This release aligns the supplied research training script with the current manuscript's method and protocol. It changes architecture and supervision relative to that script. Original checkpoints are **not compatible**. This repository does not claim that these changes reproduce the manuscript's existing numerical results; that requires retraining and evaluation with the original private cohort and exact partitions. No clinical images, patient identifiers, trained checkpoints, or private experiment outputs are distributed.

## Installation

Use Python 3.10 or newer. Real MambaVision execution needs the Mamba CUDA extension; Linux with an NVIDIA GPU (or a compatible WSL2 CUDA environment) is the intended training environment. Install a matching CUDA-enabled PyTorch/torchvision pair following [PyTorch's instructions](https://pytorch.org/get-started/locally/), then:

```bash
git clone https://github.com/gggg123445/POS-mamba.git
cd POS-mamba
python -m pip install -e .
python -m pip install mamba-ssm==2.2.4 --no-build-isolation
python -m pos_mamba --help
```

The dependency version follows [NVIDIA's upstream requirements](https://github.com/NVlabs/MambaVision/blob/main/requirements.txt). Extension compatibility depends on the CUDA/PyTorch/compiler combination; the local checks described below did not test the extension. The backbone is downloaded through the [official Hugging Face model](https://huggingface.co/nvidia/MambaVision-T-1K). Its custom Python code is loaded with `trust_remote_code=True`. For a reproducible run, pass `--model_revision <reviewed-model-commit>` to pin both the weights and remote code. No alternative download mirror is imposed.

## Input data

The CSV contains one row per analyzed vessel:

```csv
record_id,patient_id,stenosis_percentage
SYN00001_LAD,SYN00001,12
SYN00001_RCA,SYN00001,60
```

These identifiers are illustrative. `record_id` must be unique and match image filenames. `patient_id` must identify the same patient across all vessels and examinations; specify it using `--group_col patient_id`. `uniq_ID` is accepted instead of `record_id`. Automatic grouping accepts explicit patient columns or `check_id`; it does not infer patients from arbitrary vessel filenames. If examination IDs differ for the same patient, use a genuine patient column.

Images reside directly inside `--image_root`, for example:

```text
images/
  SYN00001_LAD_cpr_view0.png
  SYN00001_LAD_cpr_view1.png
  SYN00001_RCA_cpr_view0.png
```

PNG/JPEG files are ordered by the numeric `view` suffix. `--scan_type_filter cpr` restricts matching to the illustrated modality. Up to 30 views are sampled uniformly; fewer available views are accepted. Missing images, duplicate vessel IDs, invalid labels, and overlapping patient partitions fail explicitly.

`stenosis_percentage` accepts values in 0–100, percent strings, or ranges interpreted by their midpoint. Grades are 0 at 0%; 1 below 25%; 2 below 50%; 3 below 70%; 4 below 100%; and 5 at 100%. A class-only CSV with `mapped_label` in 0–5 can be used with `--use_hlgauss false`; probabilistic supervision requires continuous stenosis targets.

## Training and evaluation

```bash
python -m pos_mamba \
  --csv /path/to/private/metadata.csv \
  --image_root /path/to/private/images \
  --group_col patient_id \
  --test_groups_file /path/to/private/fixed_holdout.json \
  --save_path outputs/paper_run \
  --gpu 0
```

The fixed holdout file has the format `{"test_groups": ["SYN00001", "SYN00002"]}`. Unknown IDs fail instead of silently changing the cohort. Use the **same original holdout and development partitions** to compare with archived results. If the file is omitted, one approximately 20% patient-stratified holdout is generated using seed 42 and reused by all development splits. This new partition does not reconstruct the manuscript's exact 181-patient/518-vessel holdout.

Three stratified development partitions are used by default, with each held-out development fold serving as validation. `--val_ratio` optionally moves extra training groups into validation; the default is zero. To replay exact archived partitions, also pass `--development_splits_file /path/to/private/development_groups.json`, formatted as `{"splits": [{"train_groups": ["..."], "val_groups": ["..."]}, ...]}`. The number of entries must match `--n_splits`; every entry must partition the entire development cohort and exclude the holdout. All checkpoints and early stopping use validation T1 macro-F1 by default. After every development split finishes, the primary split is selected by its best-checkpoint validation T1 macro-F1, with the earliest split winning a tie. `primary_selection.json` is saved **before holdout inference**. The selected split is not hardcoded to split 2; that was an observed outcome of the manuscript's runs.

| Setting | Default |
|---|---|
| Backbone | MambaVision-T-1K, final spatial stage via `AutoModel` |
| CPR representation | center crop width 40, transpose, concatenate along width; no resize/padding by default |
| Frequency branch | four-level Haar, LH/HL/HH at each level plus final LL, band attention |
| Fusion | single-channel pixel gate, broadcast across feature channels, then global average pooling |
| Optimizer | AdamW, learning rate 0.0001, weight decay 0.001 |
| Batch / epochs / schedule | 4 / 100 / cosine annealing |
| Early stopping / seed | 20 epochs / 42 |
| Probabilistic supervision | Gaussian sigma 10 percentage points, weight 0.9 |

Batch collation center-crops each long image to the smallest height and width in that batch. It introduces no padding, but can remove peripheral content from longer images. Keep batch size/order consistent when comparing outputs. Height resizing flags remain available as explicit preprocessing alternatives and are recorded in run metadata. This entry point trains on one logical GPU per process; the paper's two-RTX-5090 hardware description is not a claim of distributed training support in this release.

Task definitions:

| Head | Target | Eligible ground-truth grades |
|---|---|---|
| T1 | six grades 0–5 | all |
| T2 | grade ≥3 | all |
| T3 | 0 / {1,2} / {3,4} / 5 | all |
| T4 | 0 versus {1,2} | ≤2 |
| T5 | {3,4} versus 5 | ≥3 |
| T6 | 3 / 4 / 5 | ≥3 |

T4–T6 are trained and evaluated on reference-defined eligible subsets. These scores describe conditional performance; the implementation does not route cases using predicted categories. CE/BCE losses are averaged within each eligible subset and summed. Empty tasks contribute zero. The joint objective is `0.1 * main_loss + 0.9 * histogram_soft_CE`.

Gaussian targets integrate probability between histogram edges, with both exterior tails absorbed into the endpoint bins. The six-bin auxiliary head retains the supplied script's integer-percentage convention: edges `[0, 0.5, 24.5, 49.5, 69.5, 99.5, 100.5]`. Class labels still use the clinical cutoffs stated above. Sigma is a fixed supervision hyperparameter; auxiliary probabilities are not calibrated case-specific uncertainty estimates. `--hl_bins_mode uniform101` is an alternative, not the default.

## Outputs and checks

Outputs include run arguments, private partition manifests, histories, best checkpoints, six vessel prediction CSVs per split, patient predictions, per-split metrics, `primary_selection.json`, `primary_metrics.json`, and `summary.json`. Primary metrics correspond to the validation-selected checkpoint. Supplementary means and **sample** standard deviations describe the three checkpoints evaluated on the **same** holdout.

All task F1 scores are macro-averaged over their defined classes, including binary heads. Binary sensitivity/PPV/NPV use class 1; multiclass values use macro one-versus-rest. AUC uses predicted probabilities: positive-class probabilities for binary heads and macro one-versus-rest for multiclass heads. AUC is undefined if any required reference class is absent (missing CSV value / JSON `null`); no artificial value is substituted. T1 kappa uses linear weights. Patient T1/T2 predictions take the maximum prediction among each patient's available vessels, without inventing missing vessels.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python examples/make_synthetic_data.py --output data/synthetic
```

Tests cover spatial-gate broadcasting, Gaussian endpoint masses, conditional loss masks, gradients in both branches, SACR orientation, macro-F1/AUC, patient aggregation, and a complete three-split synthetic training/evaluation/export run. The backbone is replaced by a tiny convolution **only in tests**, so they run without model downloads or Mamba CUDA kernels. Local checks used Python 3.9.20, torch 2.8.0+cu126, torchvision 0.23.0+cu126, transformers 4.47.1, sklearn 1.6.1, and numpy 2.0.2. They do not establish full-backbone training compatibility or reproduce the paper's scores.

The old long training filename remains as a launcher. Core code is separated into `data`, `models`, `losses`, `engine`, `metrics`, `validation`, and `cli` modules. Generated data, identifiers, manifests, weights, and outputs are ignored by Git.

## License

Project-owned code is released under [MIT](LICENSE). MambaVision code/weights keep their [NVIDIA Source Code License-NC](https://github.com/NVlabs/MambaVision/blob/main/LICENSE); the project MIT license does not replace upstream terms. See [third-party notices](THIRD_PARTY_NOTICES.md).
