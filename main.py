from __future__ import annotations
from abc import ABC, abstractmethod
from typing import DefaultDict, Dict, List, Optional
from dataclasses import dataclass, asdict
import json

import torch
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
        self.predictions: Dict[str, List[PredictionResult]] = DefaultDict(list)

    def add_results(self, method: str, results: List[PredictionResult]):
        self.predictions[method] = results

    def save_results(self, path: str):
        data = {method: [asdict(r) for r in results]
                for method, results in self.predictions.items()}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_results(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        for method, results in data.items():
            self.predictions[method] = [PredictionResult(**r) for r in results]

    def summary(self):
        pass

    def plot_pareto(self):
        pass


class SimpleForward(Predictor):
    def __init__(self, hf_model, device, k: int = 30, n: int = 20):
        super().__init__(hf_model, device)
        self.k = k  # prefix length
        self.n = n  # tokens to predict

    def predict(self, example: Dict) -> PredictionResult:
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
        return f'SimpleForward(k={self.k}, n={self.n})'


def process_data(dataset: DatasetDict, tokenizer: PreTrainedTokenizer, split: str) -> Dataset:
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=512)

    tokenized_ds: Dataset = dataset[split].map(tokenize, batched=True)
    tokenized_ds.set_format("torch")
    return tokenized_ds


if __name__ == "__main__":
    device = 'cuda'
    model_str = "allegrolab/hubble-1b-100b_toks-perturbed-hf"
    model = AutoModelForCausalLM.from_pretrained(model_str)
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    ds: DatasetDict = load_dataset('allegrolab/passages_wikipedia')
    tokenized_ds = process_data(ds, tokenizer, split='train')

    predictor = SimpleForward(hf_model=model, device=device)
    r = predictor.predict(tokenized_ds[-100])

    print(tokenized_ds[-1]['text'])
    print(r)

    results = [r]
    evaluator = Evaluator()
    evaluator.add_results(predictor.name, results)
    evaluator.summary()
    evaluator.plot_pareto()
