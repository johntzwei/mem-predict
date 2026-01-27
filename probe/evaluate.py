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
from datasets.dataset_dict import DatasetDict
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from train import (
    ClassificationHeadProbe,
    IntermediateLayerProbe,
    LinearProbe,
    ProbeConfig,
    create_probe,
)


@dataclass
class EvaluationConfig:
    """Configuration for a probe evaluation."""
    name: str
    model: str
    dataset: str
    labels_path: str
    ckpt_path: str
    device: str = "cuda"
    split: str = "train"
    output_dir: str = "results/probe/eval"
    probe: ProbeConfig = field(default_factory=lambda: ProbeConfig(type="ClassificationHeadProbe"))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvaluationConfig":
        probe_dict = d.pop("probe", {})
        probe = ProbeConfig(**probe_dict)
        return cls(**d, probe=probe)

    @classmethod
    def from_json(cls, path: str) -> "EvaluationConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


def load_probe_weights(probe, ckpt_path: str, device: str = "cuda"):
    """Load probe weights from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location=device)
    probe_state = ckpt["probe_state"]
    probe_type = ckpt["probe_type"]

    if probe_type == "IntermediateLayerProbe":
        probe.probe_model.classifier.load_state_dict(probe_state)
    else:
        probe.model.load_state_dict(probe_state, strict=False)

    print(f"Loaded {probe_type} weights from {ckpt_path}")
    print(f"  Best fold: {ckpt['best_fold'] + 1}, AUROC: {ckpt['best_auroc']:.4f}")
    return probe


def run_evaluation(config: EvaluationConfig) -> Dict[str, Any]:
    """Run probe evaluation with a pre-trained checkpoint."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from predict.train import process_data

    print(f"Running evaluation: {config.name}")
    print(f"Model: {config.model}")
    print(f"Dataset: {config.dataset}")
    print(f"Checkpoint: {config.ckpt_path}")

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    tokenizer.pad_token = tokenizer.eos_token
    ds: DatasetDict = load_dataset(config.dataset)
    tokenized_ds = process_data(ds, tokenizer, split=config.split)

    with open(config.labels_path) as f:
        results = json.load(f)
    y_all = np.array([r["ground_truth"] for r in results])

    print(f"Total: {len(y_all)} examples, {y_all.sum()} extractable")

    if config.probe.type == "IntermediateLayerProbe":
        model = AutoModelForCausalLM.from_pretrained(
            config.model,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model,
            num_labels=2,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Create probe and load pre-trained weights
    probe = create_probe(model, config)
    probe = load_probe_weights(probe, config.ckpt_path, config.device)

    # Run inference on the full dataset
    print(f"\nEvaluating on {len(tokenized_ds)} examples...")
    test_proba = probe.predict_proba(tokenized_ds)

    # Calculate evaluation metrics
    auroc = roc_auc_score(y_all, test_proba)
    ap = average_precision_score(y_all, test_proba)
    f1 = f1_score(y_all, test_proba > 0.5, zero_division=0)
    print(f"\n{config.name}: AUROC={auroc:.4f}, AP={ap:.4f}, F1={f1:.4f}")

    # Save results
    output_path = Path(config.output_dir) / f"{config.name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "config": {
            "name": config.name,
            "model": config.model,
            "dataset": config.dataset,
            "ckpt_path": config.ckpt_path,
            "probe": {
                "type": config.probe.type,
                "k": config.probe.k,
                "n": config.probe.n,
                "l": config.probe.l,
            },
        },
        "metrics": {"auroc": auroc, "ap": ap, "f1": f1},
        "predictions": test_proba.tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {output_path}")

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate probe with pre-trained weights")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file",
    )
    args = parser.parse_args()

    config = EvaluationConfig.from_json(args.config)
    run_evaluation(config)
