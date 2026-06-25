"""Hyperparameter search with Optuna (TPE sampler + ASHA pruner).

Searches over LR, weight decay, neck hidden, dropout, label smoothing,
augmentation strength, batch size, grad-accum steps.

Usage:
    pip install optuna
    python v2/scripts/hyperparam_search.py \
        --config v2/configs/crnn_v2.yaml \
        --data-root /path/ocr_dataset \
        --n-trials 30 --epochs-per-trial 12 \
        --study-name vn_alpr_v2_search \
        --storage sqlite:///hp_search.db

Tip: epochs-per-trial=12 is enough — ASHA prunes losers around epoch 5.
"""
from __future__ import annotations
import argparse
import copy
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import v2  # noqa: F401
from v2.src.training.trainer_v2 import TrainerV2


def make_objective(base_cfg, args):
    import optuna

    def objective(trial: "optuna.Trial") -> float:
        cfg = copy.deepcopy(base_cfg)
        cfg["train"]["epochs"] = args.epochs_per_trial
        cfg["train"]["early_stop_patience"] = max(4, args.epochs_per_trial // 3)
        cfg["output"]["dir"] = os.path.join(args.out_root, f"trial_{trial.number:03d}")

        # ---- Search space -----------------------------------------
        cfg["optim"]["lr"] = trial.suggest_float("lr", 5e-4, 4e-3, log=True)
        cfg["optim"]["weight_decay"] = trial.suggest_float("wd", 1e-5, 1e-3, log=True)
        cfg["optim"]["betas"] = [0.9, trial.suggest_float("beta2", 0.95, 0.999)]
        cfg["optim"]["warmup_epochs"] = trial.suggest_int("warmup", 3, 12)

        cfg["model"]["neck"]["hidden"] = trial.suggest_categorical(
            "neck_hidden", [192, 256, 320, 384]
        )
        cfg["model"]["neck"]["dropout"] = trial.suggest_float("neck_dropout", 0.1, 0.4)
        cfg["model"]["head"]["dropout"] = trial.suggest_float("head_dropout", 0.05, 0.3)
        cfg["model"]["backbone"]["stochastic_depth"] = trial.suggest_float("sd", 0.0, 0.2)

        cfg["loss"]["ctc"]["entropy_weight"] = trial.suggest_float("entropy", 0.0, 0.04)
        cfg["loss"]["ctc"]["label_smoothing"] = trial.suggest_float("label_smooth", 0.0, 0.10)

        cfg["augment"]["preset"] = trial.suggest_categorical(
            "aug_preset", ["medium_ocr", "heavy_ocr"]
        )
        cfg["augment"]["random_erasing_prob"] = trial.suggest_float("erase_p", 0.0, 0.35)

        cfg["train"]["batch_size"] = trial.suggest_categorical("bs", [48, 64, 96])
        cfg["train"]["grad_accum_steps"] = trial.suggest_categorical("accum", [1, 2, 4])

        # ---- Train ------------------------------------------------
        trainer = TrainerV2(cfg)

        # patch trainer.train() to report intermediate values for pruning
        original_validate = trainer.validate
        epoch_counter = {"i": 0}

        def patched_validate():
            metrics, conf = original_validate()
            trial.report(metrics.cer, step=epoch_counter["i"])
            epoch_counter["i"] += 1
            if trial.should_prune():
                raise optuna.TrialPruned()
            return metrics, conf

        trainer.validate = patched_validate
        trainer.train()
        return trainer.best_cer

    return objective


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--epochs-per-trial", type=int, default=12)
    p.add_argument("--study-name", default="vn_alpr_v2")
    p.add_argument("--storage", default=None)
    p.add_argument("--out-root", default="runs/optuna")
    args = p.parse_args()

    import optuna

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    base_cfg["data"]["root"] = args.data_root

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=3, reduction_factor=3,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler, pruner=pruner,
        direction="minimize", load_if_exists=True,
    )
    study.optimize(
        make_objective(base_cfg, args), n_trials=args.n_trials,
        gc_after_trial=True, show_progress_bar=True,
    )
    print("Best CER:", study.best_value)
    print("Best params:", study.best_params)
    # Save best params
    with open(os.path.join(args.out_root, "best_params.yaml"), "w") as f:
        yaml.safe_dump({"best_cer": study.best_value,
                        "best_params": study.best_params}, f)


if __name__ == "__main__":
    main()
