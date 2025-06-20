# Experiment Logging and Output Directory Structure

This document explains how training and experiment outputs are organized and how to sync them with Google Cloud Storage (GCS).

---

## Output Directory Layout

Each training or hyperparameter optimization run generates a unique Hydra output directory, which serves as the root for all logs, checkpoints, and artifacts of that run.

Within this root directory, outputs are organized by fold for walk-forward validation.

Example structure:

```
<Hydra_Run_Dir>/
│
├── fold_0/
│   ├── best_model.pt           # Best model checkpoint for fold 0
│   ├── scaler_fold_0.joblib    # Scaler fitted on training data for fold 0
│   ├── training_info.json      # Per-episode and validation metrics for fold 0
│   └── other logs and artifacts
│
├── fold_1/
│   ├── best_model.pt
│   ├── scaler_fold_1.joblib
│   ├── training_info.json
│   └── ...
│
└── ...
```

The root Hydra directory is configured to be unique per run (with timestamps) and is managed automatically by Hydra. It is available via the environment variable `HYDRA_OUTPUT_DIR` during execution.

---

## Configuring Save Paths

The Trainer class anchors all outputs under this Hydra run directory automatically.

You can also set the base output directory explicitly in the Hydra config (`save_path`), but it will be resolved relative to the Hydra output directory if present.

---

## Syncing Outputs to GCS

Your Google Cloud Storage bucket named `btrading` can store all outputs safely.

To sync a completed run’s outputs to GCS, run the following command from the shell, assuming you have authenticated gcloud CLI or set up a service account:

```bash
gsutil -m rsync -r <Hydra_Run_Dir> gs://btrading/experiments/<Run_ID>/
```

For example, if your Hydra output directory is:

```
/path/to/experiments/optuna_20230701_153012/
```

You can sync it via:

```bash
gsutil -m rsync -r /path/to/experiments/optuna_20230701_153012 gs://btrading/experiments/optuna_20230701_153012/
```

This preserves directory structure and uploads all model checkpoints, scaler files, and logs for remote storage and analysis.

---

## Notes

- Ensure that the GCS bucket permissions allow write access for your account or service.
- This setup works seamlessly with your training and Optuna workflows due to unified output directory anchoring.
- You can automate syncing as part of your training or deployment pipelines.
- To retrieve outputs, simply reverse the sync:

```bash
gsutil -m rsync -r gs://btrading/experiments/optuna_20230701_153012 /local/path/to/restore/
```

---

By following this structure and syncing method, you ensure full traceability, reproducibility, and safe archival of all your reinforcement learning experiment outputs.
