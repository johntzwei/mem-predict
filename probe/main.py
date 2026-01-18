from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)


class Probe(ABC):
    """Base class for memorization probes."""

    def __init__(self, model: PreTrainedModel, device: str = "cuda") -> None:
        self.model = model
        self.device = device
        self.model.to(device)
        self._is_fitted = False

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
        self.model.train()
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
        )
        criterion = nn.CrossEntropyLoss()

        n = len(labels)
        indices = np.arange(n)

        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            total_loss = 0.0
            preds = []
            targets = []

            for idx in indices:
                input_ids = (
                    train_ds[int(idx)]["input_ids"][: self.k + self.n]
                    .unsqueeze(0)
                    .to(self.device)
                )
                label = torch.tensor(
                    [int(labels[idx])], dtype=torch.long, device=self.device
                )

                optimizer.zero_grad()
                outputs = self.model(input_ids)
                loss = criterion(outputs.logits, label)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds.append(outputs.logits.argmax(dim=-1).item())
                targets.append(int(labels[idx]))

            avg_loss = total_loss / n
            f1 = f1_score(targets, preds, zero_division=0)
            print(
                f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}, F1: {f1:.4f}"
            )

        self._is_fitted = True
        self.model.eval()

    def predict_proba(self, ds: Dataset) -> np.ndarray:
        self.model.eval()
        probas = []

        n = len(ds)
        with torch.no_grad():
            for i in range(n):
                input_ids = (
                    ds[i]["input_ids"][: self.k + self.n].unsqueeze(0).to(self.device)
                )
                outputs = self.model(input_ids)
                prob = torch.softmax(outputs.logits, dim=-1)[0, 1].cpu().numpy()
                probas.append(prob)

        return np.array(probas)

    @property
    def name(self) -> str:
        return f"ClassificationHeadProbe_k{self.k}"


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "predict"))
    from main import PredictionResult, Evaluator, process_data

    device = "cuda"
    model_str = "allegrolab/hubble-1b-100b_toks-perturbed-hf"
    model = AutoModelForSequenceClassification.from_pretrained(model_str, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    ds: DatasetDict = load_dataset("allegrolab/passages_wikipedia")
    tokenized_ds = process_data(ds, tokenizer, split="train")

    results = json.load(
        open(
            "/project2/robinjia_875/jtwei/relative_memorization/mem-predict/predict/results/wikipedia_passages/1b_100b_perturbed/SimpleForward_k30_n20.json"
        )
    )
    y_all = np.array([r["ground_truth"] for r in results])

    # Train/test split: alternating indices preserves dup count distribution
    train_idx = np.arange(0, len(y_all), 2)
    train_ds = tokenized_ds.select(train_idx.tolist())

    test_idx = np.arange(1, len(y_all), 2)
    test_ds = tokenized_ds.select(test_idx.tolist())
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    print(f"Train: {len(y_train)} examples, {y_train.sum()} extractable")
    print(f"Test:  {len(y_test)} examples, {y_test.sum()} extractable")

    probe = ClassificationHeadProbe(model, device=device, k=30, n=20, epochs=4)
    probe.fit(train_ds, y_train)

    # Convert to PredictionResult for Evaluator
    def to_prediction_results(
        proba: np.ndarray, labels: np.ndarray
    ) -> List[PredictionResult]:
        return [
            PredictionResult(
                tokens_generated=0,
                total_tokens=0,
                prediction=float(p),
                ground_truth=bool(y),
            )
            for p, y in zip(proba, labels)
        ]

    evaluator = Evaluator()

    train_proba = probe.predict_proba(train_ds)
    evaluator.add_results(
        f"{probe.name}_train", to_prediction_results(train_proba, y_train)
    )

    test_proba = probe.predict_proba(test_ds)
    evaluator.add_results(
        f"{probe.name}_test", to_prediction_results(test_proba, y_test)
    )

    print("\n=== Results ===")
    print(evaluator.summary())
