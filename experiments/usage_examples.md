# Usage Examples for Training and Hyperparameter Optimization

This document provides example commands for running training, resuming training from checkpoints, and launching hyperparameter optimization using the CLI interface.

---

## 1. Running a Single Training Run

Run the training on your dataset with default configurations:

```bash
python train.py train
```

Override specific Hydra config values from the command line, e.g., change model embedding dimension and learning rate:

```bash
python train.py train --override model.embedding_dim=512 ppo.lr=0.0001
```

---

## 2. Resuming Training from a Checkpoint

Resume training from a previously saved checkpoint file:

```bash
python train.py train --override train.checkpoint_path=outputs/optuna_20230701_153012/fold_0/best_model.pt
```

This will load model, optimizer, scheduler, and training history from the checkpoint and continue training seamlessly.

---

## 3. Running Hyperparameter Optimization (Optuna)

Run an Optuna hyperparameter search with 50 trials, saving results in the default SQLite DB and output directory:

```bash
python train.py optimize --trials 50
```

Specify a custom study name and output directory:

```bash
python train.py optimize --trials 50 --study-name my_custom_study --output-dir my_optuna_outputs
```

Change the Optuna SQLite database location:

```bash
python train.py optimize --trials 50 --db-path ./my_study.db
```

---

## Notes

- All outputs, including Hydra configs, logs, models, and checkpoint files, are saved in Hydra-managed unique output directories.
- Use `--override` on the training command for all flexible configuration changes.
- Checkpoints use the `train.checkpoint_path` config key to specify resume points.
- Hyperparameter optimization uses the objective function integrated with Hydra config, so config overrides can be managed similarly.

---

This collection helps you start quickly with training and tuning while ensuring reproducibility and seamless checkpointing support.