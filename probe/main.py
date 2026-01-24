from __future__ import annotations

import argparse
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
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


class Probe(ABC):
    """Base class for memorization probes."""

    def __init__(self, model: PreTrainedModel, device: str = "cuda") -> None:
        self.model = model
        self.device = device
        # self.model.to(device)

    @abstractmethod
    def fit(self, train_ds: Dataset, labels: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict_proba(self, ds: Dataset) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class ClassificationHeadProbe(Probe):
    """Sequence classification probe using AutoModelForSequenceClassification."""

    def __init__(
        self,
        model: PreTrainedModel,
        device: str = "cuda",
        k: int = 30,
        n: int = 20,
        lr: float = 1e-3,
        epochs: int = 4,
        batch_size: int = 8,
    ) -> None:
        Probe.__init__(self, model, device)

        self.k = k
        self.n = n
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        # Freeze base model, train only the classifier head
        for name, param in self.model.named_parameters():
            if "score" not in name:  # score is the classification head
                param.requires_grad = False

    def fit(self, train_ds: Dataset, labels: np.ndarray) -> None:
        # Prepare dataset with labels and truncated input_ids
        def preprocess(example: dict, idx: int) -> dict:
            return {
                "input_ids": example["input_ids"][: self.k + self.n],
                "labels": int(labels[idx]),
            }

        train_ds = train_ds.map(preprocess, with_indices=True)
        train_ds.set_format("torch", columns=["input_ids", "labels"])

        training_args = TrainingArguments(
            output_dir="./probe_output",
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.lr,
            logging_steps=50,
            save_strategy="no",
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
        )
        trainer.train()

    def predict_proba(self, ds: Dataset) -> np.ndarray:
        self.model.eval()
        probas = []

        with torch.no_grad():
            for i in tqdm(range(len(ds)), desc="Predicting"):
                input_ids = ds[i]["input_ids"][: self.k + self.n].unsqueeze(0)
                outputs = self.model(input_ids.to(self.model.device))
                prob = torch.softmax(outputs.logits, dim=-1)[0, 1].cpu().numpy()
                probas.append(prob)

        return np.array(probas)

    @property
    def name(self) -> str:
        return f"ClassificationHeadProbe_k{self.k}"

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels)
    
    def forward(self, hidden_states: torch.Tensor):
        # only use the last tokens hidden state for classification
        return self.classifier(hidden_states[:, -1, :])


class IntermediateLayerProbe(Probe):
    def __init__(
        self,
        model: PreTrainedModel,
        device: str = "cuda",
        k: int = 30,
        n: int = 20,
        l: int = 15,
        lr: float = 1e-3,
        epochs: int = 4,
        batch_size: int = 8,
    ) -> None:
        Probe.__init__(self, model, device)

        self.k = k
        self.n = n
        self.l = l
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.l < len(model.model.layers), "Must be a valid layer in the pretrained model"

        # create the probe manually
        hidden_dim = model.config.hidden_size
        self.classifier_probe = LinearProbe(hidden_dim, num_labels=2).to(device)

        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False

    def _extract_hidden_states(self, input_ids):
        with torch.no_grad():
            output = self.model(input_ids, output_hidden_states=True)

        return output.hidden_states[self.l + 1]

    def fit(self, train_ds: Dataset, labels: np.ndarray) -> None:
        print(f"Extacting hidden states from layer {self.l}...")
        self.model.eval()
        all_hidden = []

        # get the hidden states from original model
        for i in tqdm(range(len(train_ds)), desc="Extracting hidden states"):
            input_ids = train_ds[i]["input_ids"][: self.k + self.n].unsqueeze(0).to(self.model.device)
            hidden = self._extract_hidden_states(input_ids)
            all_hidden.append(hidden[:, -1, :].cpu())   # get the last token and move onto the cpu

        X = torch.cat(all_hidden, dim=0)
        y = torch.tensor(labels, dtype=torch.long)

        # Train the linear probe
        print(f"Training the linear probe (hidden_size={X.shape[-1]})")

        self.classifier_probe.train()
        optimizer = torch.optim.Adam(self.classifier_probe.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True) 

        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()

                output = self.classifier_probe(batch_X.unsqueeze(1))   # batch, seq_len, hidden_dim
                loss = criterion(output, batch_y)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(loader):.4f}")

    def predict_proba(self, ds: Dataset) -> np.ndarray:
        self.model.eval()
        self.classifier_probe.eval()
        probas = []

        with torch.no_grad():
            for i in tqdm(range(len(ds)), desc="Predicting"):
                input_ids = ds[i]["input_ids"][: self.k + self.n].unsqueeze(0).to(self.model.device)
                hidden = self._extract_hidden_states(input_ids)
                # hidden is (1, seq_len, hidden_dim), take last token
                hidden_last = hidden[:, -1, :].unsqueeze(1)  # (1, 1, hidden_dim)
                logits = self.classifier_probe(hidden_last)  # (1, 2)
                prob = torch.softmax(logits, dim=-1)[0, 1].cpu().numpy()
                probas.append(prob)

        return np.array(probas)

    @property
    def name(self) -> str:
        return f"IntermediateLayerProbe_k{self.k}_l{self.l}"

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
        )
    else:
        raise ValueError(f"Unknown probe type: {probe_config.type}")


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run a probe experiment with the given config."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "predict"))
    from main import process_data

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

    for fold, (train_idx, test_idx) in enumerate(kfold.split(y_all)):
        print(f"\n=== Fold {fold + 1}/{config.n_splits} ===")

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

    # Overall metrics
    print("\n=== Overall ===")
    overall_metrics = evaluate(y_all, all_proba, "Overall")

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
            },
            "n_splits": config.n_splits,
        },
        "fold_results": fold_results,
        "overall": overall_metrics,
        "predictions": all_proba.tolist(),
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
