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

Accepted error uses accepted known records, as in the manuscript objective; accepted unknowns have a separate `accepted_unknown_rate`. Readable-known coverage and per-class retention use readable known records; semantic AUROC uses readable known and unknown records. Class eligibility is frozen from calibration support (`retention_min_support`, default 5), saved in the policy and used in evaluation. Missing test support leaves the full minimum-retention endpoint undefined; `class_retention.csv` exposes the individual supports.

```bash
python -m qualitynovelty.validate --output runs/validation
```

Seed 13, CPU thread 1: preparation, one training epoch, novelty fit, feasible constrained calibration, test inference, evaluation and routing plot passed. The joint gradient check loss was finite (1.3831513), classifier and technical gradients were finite, the classifier updated, and save/load maximum output difference was 0. A hand fixture verifies known-only accepted error 0 despite one accepted unknown, unknown acceptance 0.5, semantic AUROC 1, minimum retention 1 and wrong-action fraction 0.25. Fixtures are not paper-performance evidence.

The compact residual width, diagonal class-distance conditioning and grid-search policy are explicit implementation choices. Technical augmentation now combines baseline drift, electrode-motion-like transients, white noise at a declared SNR and separately modeled missing leads. The waveform transformations do not imply a clinical readability label. The comparison utility aggregates actual metric files, without preset paper scores. Real clinical annotation and full paper-table reproduction have not been run.

## Native data and semantic protocol

Install `python -m pip install -e '.[native]'` for native WFDB input. The converter reads PTB-XL's `ptbxl_database.csv`, `scp_statements.csv` and waveform files, or native challenge WFDB headers with `#Dx:`/`# Dx:` comments. It reorders the declared twelve leads, converts physical units to mV and resamples to a common rate. Short recordings fail instead of receiving fabricated waveform padding. PTB-XL roles are fixed before reading rhythm labels: official folds 1--7 train, 8 development, 9 calibration, 10 test. External challenge input is test-only and requires explicit global patient identities.

Supply a JSON rhythm mapping with `class_names`, `known` (original code to contiguous class ID), `development_unknown`, `test_unknown`, and `unknown_semantics` (every held-out code to its canonical rhythm name). All classes in the configured known vocabulary need source-training support. PTB-XL codes must be marked as rhythm codes in the release; diagnostic superclasses cannot substitute for rhythms. Unknown code sets and their canonical semantics must be disjoint between development and final testing. Ambiguous simultaneous rhythm labels are excluded and logged. Use corresponding native codes but identical class names/unknown semantics for each repository. The converter does not invent a cross-repository clinical mapping.

The annotation CSV has `record,defect,readable`, where the two binary annotations have independent definitions. For PTB-XL, `record` is the original `ecg_id`; optional `patient` overrides the namespaced native identity for a verified shared patient. Challenge records additionally require `patient,split` (`split=test`). An optional `canonical_record` identifies redistributed copies of the same acquisition. Merge checks physical/semantic schemas, deduplicates these acquisitions, and rejects conflicting roles or labels. It cannot infer unprovided duplicate identities or clinical readability.

```bash
python -m qualitynovelty.convert --kind ptbxl --input data/ptb-xl --mapping data/ptb_rhythm_map.json --annotations data/ptb_quality.csv --output data/ptb.npz
python -m qualitynovelty.convert --kind challenge --input data/arrhythmia --mapping data/challenge_rhythm_map.json --annotations data/challenge_id_quality.csv --output data/external.npz
python -m qualitynovelty.protocol merge --inputs data/ptb.npz data/external.npz --output data/ecg.npz
```

Preparation fits normalization on source training records only. Training cannot use unknown rhythms. Novelty fitting uses readable source-development records; if no source semantic outliers are provided, a declared fixed energy/nearest-class-distance combination replaces supervised novelty fitting. The same outlier-access restriction must be used for every comparison. The calibration set selects the action policy; final test labels/readability only enter scoring.

## Executable comparison policies

Use the standard six stages with these additional `--variant` values:

| Variant | Actual fitted mechanism |
| --- | --- |
| `sqi-svm` | Twelve per-lead SQI/morphology/spectral features; RBF SVM classifier with source-dev probability calibration, and separately fitted defect SVM |
| `resnet-msp` | Full 1D ResNet18, stages 64/128/256/512, maximum softmax confidence |
| `temperature` | Same network and a positive scalar temperature minimizing source-dev known-class NLL |
| `ensemble` | Five independently initialized/trained ResNet18 members; mean probabilities and predictive entropy |
| `energy` | Negative log-sum-exp energy, standardized on source development |
| `openmax` | Correct-source mean activation vectors, top-20 eucos distance tails, Weibull fitting and ranked activation reallocation to an explicit unknown class |
| `cascade` | ResNet18 classifier with the separately trained SQI defect gate |
| `ssl` | Train-only two-view SimCLR NT-Xent pretraining with a nonlinear projection head, then supervised ResNet18 fine-tuning |
| `unconstrained` | Independently trained residual two-score system with technical/conditioned novelty scoring and no class-retention constraint |

Single-score neural controls route rejected records to review; SQI/cascade and the two-score controls also have defect evidence for reacquisition advice. Every policy uses the common calibration objective and coverage budget. Class-retention constraints are specific to the full method and its appropriate ablations. The source-development/SQI annotations are never drawn from external test unknowns. OpenMax uses SciPy maximum-likelihood Weibull tails rather than numerical LibMR compatibility, and requires at least two correctly classified source examples per class. The SimCLR ECG augmentation is conservative amplitude scaling plus small jitter; it is a task adaptation, not an external pretrained ECG checkpoint. These architecture/feature choices are explicit where the TriageECG paper does not fix them.

```bash
python -m qualitynovelty prepare --input data/ecg.npz --output runs/openmax --variant openmax
python -m qualitynovelty train --output runs/openmax
python -m qualitynovelty calibrate --output runs/openmax
python -m qualitynovelty run --output runs/openmax
python -m qualitynovelty evaluate --output runs/openmax
python -m qualitynovelty.protocol stress --source runs/default --output runs/snr --snr 24 18 12 6 0
python -m qualitynovelty.protocol coverage --source runs/default --output runs/coverage --targets .7 .8 .9
python -m qualitynovelty.validate_protocol --output runs/protocol-check
```

Stress loads the frozen source model, novelty fit and policy, changes only test waveforms/experimental defect indicators, and records measured SNR. Independent readable labels stay unchanged. Coverage panels reuse the trained model, refit the same source-development scorer and select each target budget on calibration records; infeasible points remain explicitly infeasible. `stratum_actions.csv` emits classify/reacquire/review counts and wrong-action counts for the four technical/semantic strata. No manuscript table values are injected into these outputs.

The protocol fixture runs all nine actual comparison fits, five ensemble members, source-only novelty, explicit native SCP and challenge waveform conversion, positive temperature, real Weibull tails, checkpoint prediction equality, fixed-source SNR and coverage panels. All functional data are generated; no clinical performance is asserted.
