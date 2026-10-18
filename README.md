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

Source inspection: Python 3.13.5, numpy 2.1.3, scipy 1.15.3, pandas 2.2.3, scikit-learn 1.6.1, matplotlib 3.10.0, torch 2.7.1+cu118 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training or full evaluation was run. The GitHub Actions workflow checks source syntax and package installation plus undefined-name/import linting.
