from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)

from probe import (
    ClassificationHeadProbe,
    IntermediateLayerProbe,
    Probe,
)


@dataclass
class ProbeConfig:
    """Configuration for a probe type."""
    type: str  # e.g., "ClassificationHeadProbe"
    k: int = 30
    n: int = 20
    l: int = 15
    lr: float = 1e-3
    epochs: int = 4
    batch_size: int = 8
    pooling: bool = False
    attn_weighting: bool = False


@dataclass
class ExperimentConfig:
    """Configuration for a probe experiment."""
    name: str
    model: str
    dataset: str
    labels_path: str
    device: str = "cuda"
    split: str = "train"
    n_splits: int = 3
    random_state: int = 42
    output_dir: str = "results"
    ckpt_dir: str = "ckpts"
    probe: ProbeConfig = field(default_factory=lambda: ProbeConfig(type="ClassificationHeadProbe"))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        probe_dict = d.pop("probe", {})
        probe = ProbeConfig(**probe_dict)
        return cls(**d, probe=probe)

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


def evaluate(y_true: np.ndarray, y_proba: np.ndarray, name: str) -> Dict[str, float]:
    """Compute and print AUROC, AP, and F1 for a set of predictions."""
    auroc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    f1 = f1_score(y_true, y_proba > 0.5, zero_division=0)
    print(f"{name}: AUROC={auroc:.4f}, AP={ap:.4f}, F1={f1:.4f}")
    return {"auroc": auroc, "ap": ap, "f1": f1}


def create_probe(
    model: PreTrainedModel, config: ExperimentConfig
) -> Probe:
    """Factory function to create a probe from config."""
    probe_config = config.probe
    if probe_config.type == "ClassificationHeadProbe":
        return ClassificationHeadProbe(
            model,
            device=config.device,
            k=probe_config.k,
            n=probe_config.n,
            lr=probe_config.lr,
            epochs=probe_config.epochs,
            batch_size=probe_config.batch_size,
        )
    elif probe_config.type == "IntermediateLayerProbe":
        return IntermediateLayerProbe(
            model,
            device=config.device,
            k=probe_config.k,
            n=probe_config.n,
            lr=probe_config.lr,
            l=probe_config.l,
            epochs=probe_config.epochs,
            batch_size=probe_config.batch_size,
            pooling=probe_config.pooling,
            attn_weighting=probe_config.attn_weighting,
        )
    else:
        raise ValueError(f"Unknown probe type: {probe_config.type}")


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run a probe experiment with the given config."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from predict.train import process_data

    print(f"Running experiment: {config.name}")
    print(f"Model: {config.model}")
    print(f"Dataset: {config.dataset}")

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    tokenizer.pad_token = tokenizer.eos_token
    ds: DatasetDict = load_dataset(config.dataset)
    tokenized_ds = process_data(ds, tokenizer, split=config.split)

    with open(config.labels_path) as f:
        results = json.load(f)
    y_all = np.array([r["ground_truth"] for r in results])

    print(f"Total: {len(y_all)} examples, {y_all.sum()} extractable")

    # K-fold cross validation
    kfold = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
    all_proba = np.zeros(len(y_all))
    fold_results = []

    # Track best fold for checkpoint saving
    best_fold = -1
    best_auroc = -1.0
    best_probe_state = None

    for fold, (train_idx, test_idx) in enumerate(kfold.split(y_all)):
        print(f"\n=== Fold {fold + 1}/{config.n_splits} ===")

        if config.probe.type == "IntermediateLayerProbe":
            model = AutoModelForCausalLM.from_pretrained(
                config.model,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation="eager" if config.probe.attn_weighting else None,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model,
                num_labels=2,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        model.config.pad_token_id = tokenizer.pad_token_id

        train_ds = tokenized_ds.select(train_idx.tolist())
        test_ds = tokenized_ds.select(test_idx.tolist())
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        print(f"Train: {len(y_train)} examples, {y_train.sum()} extractable")
        print(f"Test:  {len(y_test)} examples, {y_test.sum()} extractable")

        probe = create_probe(model, config)
        probe.fit(train_ds, y_train)

        test_proba = probe.predict_proba(test_ds)
        all_proba[test_idx] = test_proba
        metrics = evaluate(y_test, test_proba, f"Fold {fold + 1}")
        fold_results.append(metrics)

        # Track best fold by AUROC
        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            best_fold = fold
            # Save the probe state for the best fold (clone to CPU for portability)
            if config.probe.type == "IntermediateLayerProbe":
                best_probe_state = {k: v.cpu().clone() for k, v in probe.probe_model.classifier.state_dict().items()}
            else:
                best_probe_state = {k: v.cpu().clone() for k, v in model.state_dict().items() if "score" in k}

        # Clean up to free GPU memory before next fold
        del probe, model
        torch.cuda.empty_cache()

    # Overall metrics
    print("\n=== Overall ===")
    overall_metrics = evaluate(y_all, all_proba, "Overall")

    # Save best probe checkpoint
    print(f"\n=== Best Fold: {best_fold + 1} (AUROC={best_auroc:.4f}) ===")
    ckpt_path = Path(config.ckpt_dir) / f"{config.name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "probe_state": best_probe_state,
        "probe_type": config.probe.type,
        "best_fold": best_fold,
        "best_auroc": best_auroc,
        "config": {
            "k": config.probe.k,
            "n": config.probe.n,
            "l": config.probe.l,
            "lr": config.probe.lr,
            "epochs": config.probe.epochs,
            "batch_size": config.probe.batch_size,
            "pooling": config.probe.pooling,
            "attn_weighting": config.probe.attn_weighting,
        },
    }, ckpt_path)
    print(f"Best probe checkpoint saved to {ckpt_path}")

    # Save results
    output_path = Path(config.output_dir) / f"{config.name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "config": {
            "name": config.name,
            "model": config.model,
            "dataset": config.dataset,
            "probe": {
                "type": config.probe.type,
                "k": config.probe.k,
                "n": config.probe.n,
                "l": config.probe.l,
                "lr": config.probe.lr,
                "epochs": config.probe.epochs,
                "batch_size": config.probe.batch_size,
                "pooling": config.probe.pooling,
                "attn_weighting": config.probe.attn_weighting,
            },
            "n_splits": config.n_splits,
        },
        "fold_results": fold_results,
        "overall": overall_metrics,
        "predictions": all_proba.tolist(),
        "best_fold": best_fold,
        "best_auroc": best_auroc,
        "checkpoint_path": str(ckpt_path),
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run probe experiments")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    run_experiment(config)
