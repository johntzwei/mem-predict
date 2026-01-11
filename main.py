from __future__ import annotations

import os
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional
from dataclasses import asdict, dataclass

import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class Predictor(ABC):
    def __init__(self, hf_model: PreTrainedModel, device: str) -> None:
        self.hf_model = hf_model
        self.device = device
        self.hf_model.to(device)

    @abstractmethod
    def predict(self, example: Dict[str, Any]) -> PredictionResult:
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
    def __init__(self) -> None:
        self.predictions = defaultdict(list)

    def add_results(self, method: str, results: List[PredictionResult]) -> None:
        self.predictions[method] = results

    def save_results(self, results_dir: str, method: str) -> None:
        data = [asdict(r) for r in self.predictions[method]]
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, f'{method}.json'), 'w') as f:
            json.dump(data, f, indent=2)

    def load_results(self, results_dir: str, method: str) -> None:
        with open(os.path.join(results_dir, f'{method}.json'), 'r') as f:
            data = json.load(f)
            self.add_results(
                method,
                [PredictionResult(**r) for r in data]
            )

    def summary(self) -> pd.DataFrame:
        rows = []
        for method, preds in self.predictions.items():
            y_true = [int(p.ground_truth) for p in preds]
            y_score = [p.prediction for p in preds]
            y_pred = [int(p.prediction >= 0.5) for p in preds]

            rows.append({
                'method': method,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'auroc': roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float('nan'),
            })

        df = pd.DataFrame(rows).set_index('method')
        return df

    def plot_pareto(self, metric: str = 'auroc', save_path: Optional[str] = None) -> Figure:
        df = self.summary()

        # Get tokens processed for each method
        max_tokens = max(
            preds[0].total_tokens for preds in self.predictions.values())
        df['tokens_pct'] = [self.predictions[m]
                            [0].total_tokens / max_tokens * 100 for m in df.index]
        df = df.sort_values('tokens_pct')

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df['tokens_pct'], df[metric], 'o-', markersize=8)

        for _, row in df.iterrows():
            ax.annotate(row.name, (row['tokens_pct'], row[metric]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

        ax.set_xlabel('Tokens Processed (%)')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'Pareto Curve: Compute vs {metric.upper()}')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig


class SimpleForward(Predictor):
    def __init__(self, hf_model: PreTrainedModel, device: str, k: int = 30, n: int = 20) -> None:
        super().__init__(hf_model, device)
        self.k = k  # prefix length
        self.n = n  # tokens to predict

    def predict(self, example: Dict[str, Any]) -> PredictionResult:
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

    def __init__(self, cache_path: str, k: int = 30, n: int = 20, x: int = 5) -> None:
        self.k = k
        self.n = n
        self.x = x
        with open(cache_path, 'r') as f:
            data = json.load(f)
        self.cache = [PredictionResult(**r) for r in data]

    def _predict(self, index: int) -> PredictionResult:
        cached = self.cache[index]
        assert (cached.predicted_tokens and cached.target_tokens)

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
        return [self._predict(i) for i in range(len(self.cache))]

    @property
    def name(self) -> str:
        return f'SimpleEarlyExit_k{self.k}_n{self.n}_x{self.x}'

    def predict(self, example: Dict[str, Any]) -> PredictionResult:
        raise NotImplementedError


class CrossModelPredictor(Predictor):
    """Uses source model predictions to predict target model extractability."""

    def __init__(self, source_cache_path: str, target_cache_path: str,
                 k: int = 30, n: int = 20, x: int = 20) -> None:
        self.k = k
        self.n = n
        self.x = x
        self.source_cache = self._load_cache(source_cache_path)
        self.target_cache = self._load_cache(target_cache_path)
        assert (len(self.source_cache) == len(self.target_cache))

    def _load_cache(self, path: str) -> List[PredictionResult]:
        with open(path, 'r') as f:
            return [PredictionResult(**r) for r in json.load(f)]

    def _predict(self, index: int) -> PredictionResult:
        source = self.source_cache[index]
        target = self.target_cache[index]

        # Use first x tokens from source prediction as score
        predicted = source.predicted_tokens[:self.x]
        target_tokens = target.target_tokens
        assert (len(target_tokens) == len(predicted))

        matches = sum(p == t for p, t in zip(predicted, target_tokens))
        prediction = matches / self.x

        return PredictionResult(
            tokens_generated=self.x,
            total_tokens=self.k + self.x,
            prediction=prediction,
            ground_truth=target.ground_truth,  # from target model
            predicted_tokens=predicted,
            target_tokens=target_tokens
        )

    def predict_batch(self, ds: Dataset) -> List[PredictionResult]:
        return [self._predict(i) for i in range(len(self.source_cache))]

    @property
    def name(self) -> str:
        return f'CrossModel_k{self.k}_n{self.n}_x{self.x}'

    def predict(self, example: Dict[str, Any]) -> PredictionResult:
        raise NotImplementedError

# trick to exit early in the forward pass with a hook and exception
class EarlyExitException(Exception):
    pass

class EarlyExitForward(Predictor):
    def __init__(self, hf_model, device, k: int = 30, n: int = 20, l: int = 5):
        super().__init__(hf_model, device=device)
        self.k = k # prefix length
        self.n = n # tokens to predict
        self.l = l # layer index

        # register the hook
        self._hook_handle = self.hf_model.model.layers[l].register_forward_hook(self._hook_fn)
        self.intermediate_output = {}

    def remove_hook(self) -> None:
        """Remove the forward hook to prevent memory leaks."""
        self._hook_handle.remove()
        
    def _hook_fn(self, module, input, output):
        self.intermediate_output['output'] = output
        raise EarlyExitException()

    def _early_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.hf_model.eval()
        with torch.no_grad():
            try:
                self.hf_model.forward(input_ids=input_ids)
            except EarlyExitException:
                pass

            early_output = self.intermediate_output['output']
 
            # apply norm and lm_head to get logits from early output
            early_logits = self.hf_model.lm_head(self.hf_model.model.norm(early_output))

        return early_logits
        
    def predict(self, example) -> PredictionResult:
        input_ids = example['input_ids'][:self.k +
                                         self.n].unsqueeze(0).to(self.device)
        assert (input_ids.shape[1] == self.k + self.n)

        early_logits = self._early_forward(input_ids=input_ids)

        # logits[i] predicts token i+1; we want predictions for positions k to k+n-1
        predicted = torch.argmax(
            early_logits, dim=-1).squeeze(0)[self.k - 1: self.k + self.n - 1]
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
        return f'EarlyExitForward_k{self.k}_n{self.n}_l{self.l}'


def process_data(dataset: DatasetDict, tokenizer: PreTrainedTokenizer, split: str) -> Dataset:
    def tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(example["text"], truncation=True, max_length=512)

    tokenized_ds = dataset[split].map(tokenize, batched=True)
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
    # ===
    # 1b experiments
    # ===
    device = 'cuda'
    model_str = "allegrolab/hubble-1b-100b_toks-perturbed-hf"
    model = AutoModelForCausalLM.from_pretrained(model_str)
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    # this dataset is very imbalanced
    # the number of extractable sequences is small
    ds: DatasetDict = load_dataset('allegrolab/passages_wikipedia')
    tokenized_ds = process_data(ds, tokenizer, split='train')

    evaluator = Evaluator()
    base_results_dir = "results/wikipedia_passages/1b_100b_perturbed"

    # # SimpleForward
    # simple_forward_dir = f"{base_results_dir}/simple_forward"
    # predictor = SimpleForward(hf_model=model, device=device)
    # results = load_cached(evaluator, simple_forward_dir, predictor)

    # # SimpleEarlyExit (uses SimpleForward cache)
    # x_values = [1, 5, 10, 15, 20]
    # for x in x_values:
    #     cache_path = os.path.join(simple_forward_dir, 'SimpleForward_k30_n20.json')
    #     predictor = SimpleEarlyExit(cache_path=cache_path, x=x)
    #     results = load_cached(evaluator, simple_forward_dir, predictor)

    # EarlyExitForward
    l = 13  # layer to exit at
    early_exit_dir = f"{base_results_dir}/early_exit_l{l}"
    predictor = EarlyExitForward(hf_model=model, device=device, l=l)
    results = load_cached(evaluator, early_exit_dir, predictor)

    print(predictor.hf_model.device)

    # SimpleEarlyExit (uses EarlyExitForward cache)
    x_values = [1, 5, 10, 15, 20]
    for x in x_values:
        cache_path = os.path.join(early_exit_dir, f'EarlyExitForward_k30_n20_l{l}.json')
        predictor = SimpleEarlyExit(cache_path=cache_path, x=x)
        results = load_cached(evaluator, early_exit_dir, predictor)

    print(evaluator.summary())
    evaluator.plot_pareto(
        metric='auroc',
        save_path='results/pareto_auroc_1b.png'
    )

    # # ===
    # # 8b experiments
    # # ===
    # del (model)
    # model_str = "allegrolab/hubble-8b-100b_toks-perturbed-hf"
    # model = AutoModelForCausalLM.from_pretrained(model_str)
    # tokenizer = AutoTokenizer.from_pretrained(model_str)

    # evaluator = Evaluator()
    # results_dir = "results/wikipedia_passages/8b_100b_perturbed"

    # predictor = SimpleForward(hf_model=model, device=device)
    # results = load_cached(evaluator, results_dir, predictor)

    # x_values = [1, 5, 10, 15, 20]
    # for x in x_values:
    #     cache_path = os.path.join(results_dir, 'SimpleForward_k30_n20.json')
    #     predictor = SimpleEarlyExit(cache_path=cache_path, x=x)
    #     results = load_cached(evaluator, results_dir, predictor)

    # predictor = CrossModelPredictor(
    #     source_cache_path='results/wikipedia/1b_100b_perturbed',
    #     target_cache_path='results/wikipedia/8b_100b_perturbed'
    # )

    # print(evaluator.summary())
    # evaluator.plot_pareto(
    #     metric='auroc',
    #     save_path='results/pareto_auroc_8b.png'
    # )
