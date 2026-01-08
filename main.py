from __future__ import annotations

import os
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import asdict, dataclass

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


class Predictor(ABC):
    def __init__(self, hf_model, device):
        self.hf_model = hf_model
        self.device: str = device
        hf_model.to(device)

    @abstractmethod
    def predict(self, example) -> PredictionResult:
        pass

    def predict_batch(self, ds: Dataset) -> List[PredictionResult]:
        return [self.predict(example) for example in ds]

    @property
    @abstractmethod
    def name(self) -> str:
        pass


@dataclass
class PredictionResult:
    tokens_generated: int
    total_tokens: int
    prediction: float
    ground_truth: Optional[bool]
    predicted_tokens: Optional[List[int]] = None
    target_tokens: Optional[List[int]] = None


class Evaluator:
    def __init__(self):
        self.predictions: Dict[str, List[PredictionResult]] = defaultdict(list)

    def add_results(self, method: str, results: List[PredictionResult]):
        self.predictions[method] = results

    def save_results(self, results_dir: str, method: str):
        data = [asdict(r) for r in self.predictions[method]]
        os.makedirs(os.path.dirname(results_dir), exist_ok=True)
        with open(os.path.join(results_dir, f'{method}.json'), 'w') as f:
            json.dump(data, f, indent=2)

    def load_results(self, results_dir: str, method: str):
        with open(os.path.join(results_dir, f'{method}.json'), 'r') as f:
            data = json.load(f)
            self.add_results(
                method,
                [PredictionResult(**r) for r in data]
            )

    def compute_metrics(self, method: str) -> Dict:
        """Compute comprehensive sklearn metrics for a method's predictions."""
        results = self.predictions[method]

        # Extract predictions and ground truth
        y_pred_proba = np.array([r.prediction for r in results])
        y_true = np.array([r.ground_truth for r in results if r.ground_truth is not None])
        y_pred_proba_filtered = np.array([r.prediction for r in results if r.ground_truth is not None])

        # Binary predictions using 0.5 threshold
        y_pred_binary = (y_pred_proba_filtered >= 0.5).astype(int)

        metrics = {
            'method': method,
            'num_samples': len(results),
            'num_with_ground_truth': len(y_true),
        }

        if len(y_true) > 0:
            # Basic classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)

            # Precision, Recall, F1
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred_binary, average='binary', zero_division=0
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1

            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)

            # ROC-AUC and PR-AUC (if we have both classes)
            if len(np.unique(y_true)) > 1:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba_filtered)
                metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba_filtered)
            else:
                metrics['roc_auc'] = None
                metrics['pr_auc'] = None

            # Compute cost metrics
            avg_tokens_generated = np.mean([r.tokens_generated for r in results])
            avg_total_tokens = np.mean([r.total_tokens for r in results])
            metrics['avg_tokens_generated'] = avg_tokens_generated
            metrics['avg_total_tokens'] = avg_total_tokens
            metrics['compute_ratio'] = avg_tokens_generated / avg_total_tokens if avg_total_tokens > 0 else 0

        return metrics

    def summary(self, methods: Optional[List[str]] = None):
        """Print a summary of metrics for all or specified methods."""
        if methods is None:
            methods = list(self.predictions.keys())

        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80 + "\n")

        for method in methods:
            if method not in self.predictions:
                print(f"Warning: No results found for method '{method}'")
                continue

            metrics = self.compute_metrics(method)

            print(f"Method: {method}")
            print("-" * 80)
            print(f"  Samples: {metrics['num_samples']} (with ground truth: {metrics['num_with_ground_truth']})")

            if metrics['num_with_ground_truth'] > 0:
                print(f"\n  Classification Metrics:")
                print(f"    Accuracy:  {metrics['accuracy']:.4f}")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall:    {metrics['recall']:.4f}")
                print(f"    F1-Score:  {metrics['f1_score']:.4f}")

                if metrics['roc_auc'] is not None:
                    print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
                    print(f"    PR-AUC:    {metrics['pr_auc']:.4f}")

                print(f"\n  Confusion Matrix:")
                print(f"    TP: {metrics['true_positives']:5d}  FP: {metrics['false_positives']:5d}")
                print(f"    FN: {metrics['false_negatives']:5d}  TN: {metrics['true_negatives']:5d}")

                print(f"\n  Compute Efficiency:")
                print(f"    Avg tokens generated: {metrics['avg_tokens_generated']:.2f}")
                print(f"    Avg total tokens:     {metrics['avg_total_tokens']:.2f}")
                print(f"    Compute ratio:        {metrics['compute_ratio']:.4f}")

            print("\n")

        print("="*80 + "\n")

    def plot_pareto(self):
        pass


class SimpleForward(Predictor):
    def __init__(self, hf_model, device, k: int = 30, n: int = 20):
        super().__init__(hf_model, device)
        self.k = k  # prefix length
        self.n = n  # tokens to predict

    def predict(self, example) -> PredictionResult:
        input_ids = example['input_ids'][:self.k +
                                         self.n].unsqueeze(0).to(self.device)
        assert (input_ids.shape[1] == self.k + self.n)

        self.hf_model.eval()
        with torch.no_grad():
            logits = self.hf_model.forward(input_ids=input_ids).logits

        # logits[i] predicts token i+1; we want predictions for positions k to k+n-1
        predicted = torch.argmax(
            logits, dim=-1).squeeze(0)[self.k - 1: self.k + self.n - 1]
        target = example['input_ids'][self.k: self.k + self.n].to(self.device)

        matches = (predicted == target)
        prediction = matches.float().mean().item()
        ground_truth = matches.all().item()

        return PredictionResult(
            tokens_generated=self.n,
            total_tokens=self.k + self.n,
            prediction=prediction,
            ground_truth=ground_truth,
            predicted_tokens=predicted.tolist(),
            target_tokens=target.tolist()
        )

    @property
    def name(self) -> str:
        return f'SimpleForward_k{self.k}_n{self.n}'


class SimpleEarlyExit(Predictor):
    """Evaluates early exit predictions from cached SimpleForward results."""

    def __init__(self, cache_path: str, k: int = 30, n: int = 20, x: int = 5):
        self.k = k
        self.n = n
        self.x = x
        with open(cache_path, 'r') as f:
            data = json.load(f)
        self.cache: List[PredictionResult] = [
            PredictionResult(**r) for r in data]

    def predict(self, example) -> PredictionResult:
        raise NotImplementedError

    def _predict(self, index: int) -> PredictionResult:
        cached: PredictionResult = self.cache[index]
        predicted = cached.predicted_tokens[:self.x]
        target = cached.target_tokens[:self.x]

        matches = sum(p == t for p, t in zip(predicted, target))
        prediction = matches / self.x

        return PredictionResult(
            tokens_generated=self.x,
            total_tokens=self.k + self.x,
            prediction=prediction,
            ground_truth=cached.ground_truth,
            predicted_tokens=predicted,
            target_tokens=target
        )

    def predict_batch(self, ds: Dataset) -> List[PredictionResult]:
        assert (len(ds) == len(self.cache))
        return [self.predict(i) for i in range(len(self.cache))]

    @property
    def name(self) -> str:
        return f'SimpleEarlyExit_k{self.k}_n{self.n}_x{self.x}'


def process_data(dataset: DatasetDict, tokenizer: PreTrainedTokenizer, split: str) -> Dataset:
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=512)

    tokenized_ds: Dataset = dataset[split].map(tokenize, batched=True)
    tokenized_ds.set_format("torch")
    return tokenized_ds


def load_cached(evaluator: Evaluator, results_dir: str, predictor: Predictor) -> List[PredictionResult]:
    """Load results from cache if they exist, otherwise run and save."""
    cache_path = os.path.join(results_dir, f'{predictor.name}.json')
    if os.path.exists(cache_path):
        print(f"Loading cached results from {cache_path}")
        evaluator.load_results(results_dir, predictor.name)
    else:
        results = predictor.predict_batch(tqdm(tokenized_ds))
        evaluator.add_results(predictor.name, results)
        evaluator.save_results(results_dir, predictor.name)

    return evaluator.predictions[predictor.name]


if __name__ == "__main__":
    device = 'cuda'
    model_str = "allegrolab/hubble-1b-100b_toks-perturbed-hf"

    model = AutoModelForCausalLM.from_pretrained(model_str)
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    ds: DatasetDict = load_dataset('allegrolab/passages_wikipedia')
    tokenized_ds = process_data(ds, tokenizer, split='train')

    evaluator = Evaluator()
    results_dir = "results/wikipedia_passages/"

    # SimpleForward
    predictor = SimpleForward(hf_model=model, device=device)
    results = load_cached(evaluator, results_dir, predictor)

    # SimpleEarlyExit
    x_values = [1, 5, 10, 15, 20]
    for x in x_values:
        cache_path = os.path.join(results_dir, 'SimpleForward_k30_n20.json')
        predictor = SimpleEarlyExit(cache_path=cache_path, x=x)
        results = load_cached(evaluator, results_dir, predictor)

    # Display comprehensive metrics summary
    evaluator.summary()
