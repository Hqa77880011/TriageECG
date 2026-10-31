# TriageECG

Implementation of **Separating signal corruption from semantic novelty for class-preserving ECG triage.**

## Installation

Use Python 3.10+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Use [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) and the paper's arrhythmia sources after a declared common rhythm mapping. NPZ fields are `x: (N,leads,time)`, `y` with known classes `0..K-1` and semantic unknown `-1`, `patient`, `split`, `defect` and `readable`. Roles are `train,dev,cal,test`; entire patients stay in one role. Training contains known classes only. Development semantic outliers are source-held-out classes, never final-target unknowns. Technical-defect and readable labels describe separate annotation concepts.

```bash
python -m qualitynovelty prepare --input data/ecg.npz --config config.json --output runs/default
```

## Training and inference

```bash
python -m qualitynovelty train --config config.json --output runs/default
python -m qualitynovelty calibrate --config config.json --output runs/default
python -m qualitynovelty run --config config.json --output runs/default
```

## Evaluation

```bash
python -m qualitynovelty evaluate --config config.json --output runs/default
python -m qualitynovelty plot --config config.json --output runs/default
```

## Ablations

Supported controls: `no-quality`, `no-retention`, `no-consistency`, `collapsed-actions`. Keep the input groups, seed and budget fixed.

```bash
python -m qualitynovelty prepare --input data/ecg.npz --config config.json --output runs/no_quality --variant no-quality
python -m qualitynovelty train --config config.json --output runs/no_quality
python -m qualitynovelty calibrate --config config.json --output runs/no_quality
python -m qualitynovelty run --config config.json --output runs/no_quality
python -m qualitynovelty evaluate --config config.json --output runs/no_quality
```

## Outputs

`runs/default/` holds `settings.json`, `data.npz`, normalization/model state, `novelty.npz`, `policy.json`, `predictions.npz`, `metrics.csv`, `action_confusion.csv` and `routing.png`.

Keep `config.json` with the run; it contains model/data settings and the seed where the CLI does not expose one. Use separate output directories for different seeds/conditions.

CPU validation: Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3, pandas 2.2.3, scikit-learn 1.6.1, Matplotlib 3.10.0 and PyTorch 2.7.1+cu118. The check below actually exercises all six CLI stages with synthetic patient-disjoint data.

## Method and metric mapping

| Paper component / endpoint | Entry | Command |
| --- | --- | --- |
| Residual temporal encoder, known and technical heads, mild-pair loss | `Triage`, `train` | `python -m qualitynovelty train` |
| Source-dev energy/distance/quality novelty | `novelty_features`, `calibrate` | `python -m qualitynovelty calibrate` |
| Coverage and class-retention constrained three-action policy | `route`, `calibrate` | `calibrate` / `run` |
| Clean macro-F1, readable semantic AUROC, accepted-known error, retention and action rates | `evaluate` | `python -m qualitynovelty evaluate` |
| Routing scores | `plot` | `python -m qualitynovelty plot` |

Accepted error uses accepted known records, as in the manuscript objective; accepted unknowns have a separate `accepted_unknown_rate`. Readable-known coverage and per-class retention use readable known records; semantic AUROC uses readable known and unknown records. Class retention currently requires any calibration support; larger minimum-support eligibility must be frozen in the input protocol.

```bash
python -m qualitynovelty.validate --output runs/validation
```

Seed 13, CPU thread 1: preparation, one training epoch, novelty fit, feasible constrained calibration, test inference, evaluation and routing plot passed. The joint gradient check loss was finite (1.4880903), classifier and technical gradients were finite, the classifier updated, and save/load maximum output difference was 0. A hand fixture verifies known-only accepted error 0 despite one accepted unknown, unknown acceptance 0.5, semantic AUROC 1, minimum retention 1 and wrong-action fraction 0.25. Fixtures are not paper-performance evidence.

The compact residual width, diagonal class-distance conditioning, Gaussian technical perturbation and grid-search policy are explicit implementation choices. This implementation does not include external SQI+SVM, temperature-scaled/deep-ensemble, OpenMax or SSL backbone baseline training. The comparison utility aggregates actual metric files, without preset paper scores. Real ECG mapping, clinical technical/readability annotation and paper-table reproduction have not been run.
