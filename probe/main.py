from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)


class Probe(ABC):
    """Base class for memorization probes."""

    def __init__(self, model: PreTrainedModel, device: str = "cuda") -> None:
        self.model = model
        self.device = device
        self.model.to(device)

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
    ) -> None:
        Probe.__init__(self, model, device)

        self.k = k
        self.n = n
        self.lr = lr
        self.epochs = epochs

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
            per_device_train_batch_size=8,
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

        n = len(ds)
        with torch.no_grad():
            for i in range(n):
                input_ids = (
                    ds[i]["input_ids"][: self.k +
                                       self.n].unsqueeze(0).to(self.device)
                )
                outputs = self.model(input_ids)
                prob = torch.softmax(
                    outputs.logits, dim=-1)[0, 1].cpu().numpy()
                probas.append(prob)

        return np.array(probas)

    @property
    def name(self) -> str:
        return f"ClassificationHeadProbe_k{self.k}"


def evaluate(y_true: np.ndarray, y_proba: np.ndarray, name: str) -> None:
    """Print AUROC, AP, and F1 for a set of predictions."""
    auroc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    f1 = f1_score(y_true, y_proba > 0.5, zero_division=0)
    print(f"{name}: AUROC={auroc:.4f}, AP={ap:.4f}, F1={f1:.4f}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "predict"))
    from main import process_data

    device = "cuda"
    model_str = "allegrolab/hubble-1b-100b_toks-perturbed-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    ds: DatasetDict = load_dataset("allegrolab/passages_wikipedia")
    tokenized_ds = process_data(ds, tokenizer, split="train")

    results = json.load(
        open(
            "/project2/robinjia_875/jtwei/relative_memorization/mem-predict/predict/results/wikipedia_passages/1b_100b_perturbed/SimpleForward_k30_n20.json"
        )
    )
    y_all = np.array([r["ground_truth"] for r in results])

    print(f"Total: {len(y_all)} examples, {y_all.sum()} extractable")

    # K-fold cross validation
    n_splits = 3
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_proba = np.zeros(len(y_all))

    for fold, (train_idx, test_idx) in enumerate(kfold.split(y_all)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")

        # Reload model each fold to reset classifier head
        model = AutoModelForSequenceClassification.from_pretrained(
            model_str, num_labels=2)

        train_ds = tokenized_ds.select(train_idx.tolist())
        test_ds = tokenized_ds.select(test_idx.tolist())
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        print(f"Train: {len(y_train)} examples, {y_train.sum()} extractable")
        print(f"Test:  {len(y_test)} examples, {y_test.sum()} extractable")

        probe = ClassificationHeadProbe(
            model, device=device, k=30, n=20, epochs=2)
        probe.fit(train_ds, y_train)

        test_proba = probe.predict_proba(test_ds)
        all_proba[test_idx] = test_proba
        evaluate(y_test, test_proba, f"Fold {fold + 1}")
