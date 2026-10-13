# Alignment with the current POS-Mamba manuscript

The release follows the manuscript's SACR, frequency branch, pixel gate, clinical-prior tasks, probabilistic supervision, implementation settings, and fixed-holdout evaluation definitions. It is a corrected implementation, not a numerical equivalence claim for the earlier research script.

| Manuscript statement | Release implementation | Change from supplied script |
|---|---|---|
| MambaVision provides a spatial semantic feature map | `POSMamba.forward` uses the last `AutoModel` stage | eliminates classifier-logit/1×1 fallback and channel interpolation |
| SACR transposes cropped CPR views and concatenates without resizing | `VesselLongImageDatasetNativeNoPad` uses PIL `TRANSPOSE`, crop-only alignment | replaces 90° rotation; disables post-concatenation height interpolation by default |
| Four-level Haar includes LH, HL, HH and LL | four levels with all detail bands plus final LL | enables HH by default |
| Band attention is softmax of an MLP on pooled bands | raw-band attention by default, temperature 1 | disables optional per-band centering by default, which would collapse mean descriptors |
| Gate is sigmoid(Conv1×1(ReLU(BN(Conv3×3(concat))))) and shared over channels | `SpatialGatedFusion` outputs B×1×H×W | replaces GELU and channel-specific gate |
| Fusion precedes global pooling | spatial alignment → gate → GAP → heads | enforced; invalid spatial output fails |
| T1–T6 conditional targets and mean masked losses | `get_task_labels`, `compute_multitask_loss` | definitions retained, verified including empty subsets |
| Gaussian tails are assigned to endpoint bins | `hl_gauss_to_probs` integrates interior edges and absorbs exterior tails | replaces truncated Gaussian renormalization |
| Sigma 10, lambda 0.9 | default sigma and histogram weight | retained; described as supervision smoothing |
| Batch 4, AdamW lr 1e-4, weight decay 1e-3 | CLI defaults | replaces batch 3 and weight decay 1e-4 |
| Cosine schedule, 100 epochs, patience 20, seed 42 | training engine and CLI | retained |
| Primary checkpoint selected by validation T1 macro-F1 | per-split checkpoint selection; primary selection saved before holdout evaluation | adds explicit cross-split selection; no hardcoded split-2 result |
| Three development splits share one patient holdout | CLI partitioning and overlap checks | clarifies protocol and output names; patient identities required |
| Binary F1 is macro; AUC uses probabilities | `classification_metrics` | replaces positive-class-only F1; adds AUC |
| Patient grade is maximum over available vessels | `patient_metrics` | adds patient T1/T2 metrics and prediction export |

The manuscript does not specify every internal width, histogram edge, or development partition rule. Feature projection 512, tower hidden width 128, frequency width 256, auxiliary width 256, histogram half-integer edges, and grayscale normalization (mean/std 0.5) are documented implementation choices inherited from the supplied script. Default development splits are stratified K-fold partitions of the non-holdout cohort; `--development_splits_file` accepts exact archived train/validation groups when these are available. No new training results have been used to replace manuscript tables.

The full experimental comparison, baseline implementations, paired bootstrap/Holm analyses, and original medical dataset are not bundled in this training-code release. The reported hardware uses two RTX 5090 GPUs; this launcher executes on one selected GPU per process. A real MambaVision/CUDA training validation remains necessary because automated tests replace only the backbone with a tiny interface-compatible module.
