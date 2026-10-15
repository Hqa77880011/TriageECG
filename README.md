# TriageECG

Implementation of **Separating signal corruption from semantic novelty for class-preserving ECG triage.**

## Install

Use Python 3.11 in an isolated environment. Run from the repository root:

```bash
python -m pip install -e .
```

## Data

Use [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) and the paper's arrhythmia sources after a declared common rhythm mapping. NPZ fields are `x: (N,leads,time)`, `y` with known classes `0..K-1` and semantic unknown `-1`, `patient`, `split`, `defect` and `readable`. Roles are `train,dev,cal,test`; entire patients stay in one role. Training contains known classes only. Development semantic outliers are source-held-out classes, never final-target unknowns. Technical-defect and readable labels describe separate annotation concepts.

## Workflow

```bash
python -m qualitynovelty prepare --input data/ecg.npz
python -m qualitynovelty train
python -m qualitynovelty calibrate
python -m qualitynovelty run
python -m qualitynovelty evaluate
python -m qualitynovelty plot
```

Main commands accept `--config config.json` and `--output runs/default`. Each output directory records `settings.json`. Keep independent seeds, datasets and conditions in separate directories.

## Configuration

`technical_weight,pair_weight` combine technical and mild-pair losses. `mild_scale` defines label-preserving perturbations; `corruption_scale,lead_dropout` train technical discrimination. `coverage,coverage_tolerance,minimum_retention` define calibration feasibility. `reacquisition_cost,review_cost` price routing actions. `threshold_points` defines the finite policy family. Additional optimizer, batch-size and model-width defaults are in `config.json`.

## Outputs

`model.pt` contains the shared representation/quality model. `novelty.npz` stores source class statistics and the quality-conditioned logistic score. `policy.json` records the selected feasible threshold triple or infeasibility. `predictions.npz` retains class, defect, novelty and action outputs. `metrics.csv` contains clean Macro-F1, semantic AUROC, accepted error over every classified record, readable-known coverage, minimum class retention, routing rates and wrong-action rate; `action_confusion.csv` preserves action denominators.

## Method and comparisons

Classification, technical evidence and novelty use separate heads/inputs. Mild consistency never treats severe corruption as label preserving. Calibration enumerates a transparent three-action threshold family at the requested coverage and class-retention limits. No feasible policy produces an explicit infeasible operating point, not relaxed constraints. Acquisition/review target actions for the controlled analysis follow supplied technical/semantic labels.

Controls: `full`, `no-quality`, `no-retention`, `no-consistency`, `collapsed-actions`. Choose `--variant` and use a separate training/output directory.

```bash
python -m qualitynovelty.compare --inputs "runs/*/metrics.csv" --reference full --metrics clean_macro_f1 accepted_error minimum_class_retention --unit seed
```

Comparison averages rows within independent units before bootstrap intervals and paired Wilcoxon/Holm inference. For participant/client analyses choose those independent unit columns instead of counting recordings as replicates.

