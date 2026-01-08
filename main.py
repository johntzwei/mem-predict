from __future__ import annotations

import os
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import asdict, dataclass

import torch
from tqdm import tqdm
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


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

    def summary(self):
        pass

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
