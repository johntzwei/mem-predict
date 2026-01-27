from __future__ import annotations

import argparse
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


@dataclass
class PredictorConfig:
    """Configuration for a predictor type."""

    type: str  # e.g., "SimpleForward", "EarlyLayerExit", "EarlyTokenExit"
    k: int = 30
    n: int = 20
    l: List[int] = field(default_factory=lambda: [15])  # layers for EarlyLayerExit
    x: List[int] = field(default_factory=lambda: [5])  # tokens for EarlyTokenExit
    source_cache_path: Optional[str] = None  # for CrossModelPredictor
    target_cache_path: Optional[str] = None  # for CrossModelPredictor

    def __post_init__(self):
        # Convert single values to lists for convenience
        if isinstance(self.l, int):
            self.l = [self.l]
        if isinstance(self.x, int):
            self.x = [self.x]


@dataclass
class ExperimentConfig:
    """Configuration for a prediction experiment."""

    name: str
    model: str
    dataset: str
    device: str = "cuda"
    split: str = "train"
    output_dir: str = "results"
    plot_pareto: bool = True
    predictors: List[PredictorConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        predictors_list = d.pop("predictors", [])
        predictors = [PredictorConfig(**p) for p in predictors_list]
        return cls(**d, predictors=predictors)

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


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
        with open(os.path.join(results_dir, f"{method}.json"), "w") as f:
            json.dump(data, f, indent=2)

    def load_results(self, results_dir: str, method: str) -> None:
        with open(os.path.join(results_dir, f"{method}.json"), "r") as f:
            data = json.load(f)
            self.add_results(method, [PredictionResult(**r) for r in data])

    def summary(self) -> pd.DataFrame:
        rows = []
        for method, preds in self.predictions.items():
            y_true = [int(p.ground_truth) for p in preds]
            y_score = [p.prediction for p in preds]
            y_pred = [int(p.prediction >= 0.5) for p in preds]

            rows.append(
                {
                    "method": method,
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1": f1_score(y_true, y_pred, zero_division=0),
                    "auroc": roc_auc_score(y_true, y_score)
                    if len(set(y_true)) > 1
                    else float("nan"),
                }
            )

        df = pd.DataFrame(rows).set_index("method")
        return df

    def plot_pareto(
        self, metric: str = "auroc", save_path: Optional[str] = None
    ) -> Figure:
        df = self.summary()

        # Get tokens processed for each method
        total_tokens_by_method = [
            sum(p.total_tokens for p in preds) for preds in self.predictions.values()
        ]
        tokens_generated_per_method = [
            sum(p.tokens_generated for p in preds)
            for preds in self.predictions.values()
        ]

        df["tokens_pct"] = [
            gt / tt
            for gt, tt in zip(tokens_generated_per_method, total_tokens_by_method)
        ]
        df = df.sort_values("tokens_pct")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df["tokens_pct"], df[metric], "o", markersize=8)

        for _, row in df.iterrows():
            ax.annotate(
                row.name,
                (row["tokens_pct"], row[metric]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        ax.set_xlabel("Tokens Processed (%)")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Pareto Curve: Compute vs {metric.upper()}")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig


class SimpleForward(Predictor):
    def __init__(
        self, hf_model: PreTrainedModel, device: str, k: int = 30, n: int = 20
    ) -> None:
        super().__init__(hf_model, device)
        self.k = k  # prefix length
        self.n = n  # tokens to predict

    def predict(self, example: Dict[str, Any]) -> PredictionResult:
        input_ids = example["input_ids"][: self.k + self.n].unsqueeze(0).to(self.device)
        assert input_ids.shape[1] == self.k + self.n

        self.hf_model.eval()
        with torch.no_grad():
            logits = self.hf_model.forward(input_ids=input_ids).logits

        # logits[i] predicts token i+1; we want predictions for positions k to k+n-1
        predicted = torch.argmax(logits, dim=-1).squeeze(0)[
            self.k - 1 : self.k + self.n - 1
        ]
        target = example["input_ids"][self.k : self.k + self.n].to(self.device)

        matches = predicted == target
        prediction = matches.float().mean().item()
        ground_truth = matches.all().item()

        return PredictionResult(
            tokens_generated=self.n,
            total_tokens=self.k + self.n,
            prediction=prediction,
            ground_truth=ground_truth,
            predicted_tokens=predicted.tolist(),
            target_tokens=target.tolist(),
        )

    @property
    def name(self) -> str:
        return f"SimpleForward_k{self.k}_n{self.n}"


class EarlyLayerExit(SimpleForward):
    def __init__(
        self, hf_model: PreTrainedModel, device: str, l: int, k: int = 30, n: int = 20
    ) -> None:
        super().__init__(hf_model, device, k=k, n=n)
        self.early_layer = (
            l  # length to exit at, 15 is default since its the last layer
        )

        # register the hook
        self._hook_handle = self.hf_model.model.layers[l].register_forward_hook(
            self._hook_fn
        )
        self.intermediate_output = {}

    def remove_hook(self) -> None:
        """Remove the forward hook to prevent memory leaks."""
        self._hook_handle.remove()

    def _hook_fn(self, module, input, output):
        self.intermediate_output["output"] = output

    def predict(self, example: Dict[str, Any]) -> PredictionResult:
        sf_result = super().predict(example)

        self.hf_model.eval()
        with torch.no_grad():
            early_output = self.intermediate_output["output"]
            early_logits = self.hf_model.lm_head(self.hf_model.model.norm(early_output))

        # Compute the prediction for the early output
        target = example["input_ids"][self.k : self.k + self.n].to(self.device)
        early_predicted = torch.argmax(early_logits, dim=-1).squeeze(0)[
            self.k - 1 : self.k + self.n - 1
        ]
        early_matches = early_predicted == target

        early_prediction = early_matches.float().mean().item()
        early_predicted_tokens = early_predicted.tolist()

        return PredictionResult(
            tokens_generated=(self.k + self.n)
            * (self.early_layer / len(self.hf_model.model.layers)),
            total_tokens=self.k + self.n,
            prediction=early_prediction,
            ground_truth=sf_result.ground_truth,
            predicted_tokens=early_predicted_tokens,
            target_tokens=target.tolist(),
        )

    @property
    def name(self) -> str:
        return f"EarlyLayerExit_k{self.k}_n{self.n}_l{self.early_layer}"


class EarlyTokenExit(Predictor):
    """Evaluates early exit predictions from cached SimpleForward results."""

    def __init__(self, cache_path: str, k: int = 30, n: int = 20, x: int = 5) -> None:
        self.k = k
        self.n = n
        self.x = x
        with open(cache_path, "r") as f:
            data = json.load(f)
        self.cache = [PredictionResult(**r) for r in data]

    def _predict(self, index: int) -> PredictionResult:
        cached = self.cache[index]
        assert cached.predicted_tokens and cached.target_tokens

        predicted = cached.predicted_tokens[: self.x]
        target = cached.target_tokens[: self.x]

        matches = sum(p == t for p, t in zip(predicted, target))
        prediction = matches / self.x

        return PredictionResult(
            tokens_generated=self.x,
            total_tokens=self.k + self.x,
            prediction=prediction,
            ground_truth=cached.ground_truth,
            predicted_tokens=predicted,
            target_tokens=target,
        )

    def predict_batch(self, ds: Dataset) -> List[PredictionResult]:
        assert len(ds) == len(self.cache)
        return [self._predict(i) for i in range(len(self.cache))]

    @property
    def name(self) -> str:
        return f"EarlyTokenExit_k{self.k}_n{self.n}_x{self.x}"

    def predict(self, example: Dict[str, Any]) -> PredictionResult:
        raise NotImplementedError


class CrossModelPredictor(Predictor):
    """Uses source model predictions to predict target model extractability."""

    def __init__(
        self,
        source_cache_path: str,
        target_cache_path: str,
        k: int = 30,
        n: int = 20,
        x: int = 20,
    ) -> None:
        self.k = k
        self.n = n
        self.x = x
        self.source_cache = self._load_cache(source_cache_path)
        self.target_cache = self._load_cache(target_cache_path)
        assert len(self.source_cache) == len(self.target_cache)

    def _load_cache(self, path: str) -> List[PredictionResult]:
        with open(path, "r") as f:
            return [PredictionResult(**r) for r in json.load(f)]

    def _predict(self, index: int) -> PredictionResult:
        source = self.source_cache[index]
        target = self.target_cache[index]

        # Use first x tokens from source prediction as score
        predicted = source.predicted_tokens[: self.x]
        target_tokens = target.target_tokens

        matches = sum(p == t for p, t in zip(predicted, target_tokens))
        prediction = matches / self.x

        return PredictionResult(
            tokens_generated=self.x,
            total_tokens=self.k + self.x,
            prediction=prediction,
            ground_truth=target.ground_truth,  # from target model
            predicted_tokens=predicted,
            target_tokens=target_tokens,
        )

    def predict_batch(self, ds: Dataset) -> List[PredictionResult]:
        return [self._predict(i) for i in range(len(self.source_cache))]

    @property
    def name(self) -> str:
        return f"CrossModel_k{self.k}_n{self.n}_x{self.x}"

    def predict(self, example: Dict[str, Any]) -> PredictionResult:
        raise NotImplementedError


def process_data(
    dataset: DatasetDict, tokenizer: PreTrainedTokenizer, split: str
) -> Dataset:
    def tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(example["text"], truncation=True, max_length=512)

    tokenized_ds = dataset[split].map(tokenize, batched=True)
    tokenized_ds.set_format("torch")
    return tokenized_ds


def load_cached(
    evaluator: Evaluator,
    results_dir: str,
    predictor: Predictor,
    tokenized_ds: Dataset,
) -> List[PredictionResult]:
    """Load results from cache if they exist, otherwise run and save."""
    cache_path = os.path.join(results_dir, f"{predictor.name}.json")
    if os.path.exists(cache_path):
        print(f"Loading cached results from {cache_path}")
        evaluator.load_results(results_dir, predictor.name)
    else:
        results = predictor.predict_batch(tqdm(tokenized_ds))
        evaluator.add_results(predictor.name, results)
        evaluator.save_results(results_dir, predictor.name)

    return evaluator.predictions[predictor.name]


def create_predictors(
    pred_config: PredictorConfig,
    model: Optional[PreTrainedModel],
    device: str,
    results_dir: str,
) -> List[Predictor]:
    """Factory function to create predictors from config. Returns a list since l/x can have multiple values."""
    predictors = []

    if pred_config.type == "SimpleForward":
        assert model is not None, "SimpleForward requires a model"
        predictors.append(
            SimpleForward(
                hf_model=model,
                device=device,
                k=pred_config.k,
                n=pred_config.n,
            )
        )
    elif pred_config.type == "EarlyLayerExit":
        assert model is not None, "EarlyLayerExit requires a model"
        for l in pred_config.l:
            predictors.append(
                EarlyLayerExit(
                    hf_model=model,
                    device=device,
                    l=l,
                    k=pred_config.k,
                    n=pred_config.n,
                )
            )
    elif pred_config.type == "EarlyTokenExit":
        # EarlyTokenExit uses cached SimpleForward results
        cache_path = os.path.join(
            results_dir, f"SimpleForward_k{pred_config.k}_n{pred_config.n}.json"
        )
        for x in pred_config.x:
            predictors.append(
                EarlyTokenExit(
                    cache_path=cache_path,
                    k=pred_config.k,
                    n=pred_config.n,
                    x=x,
                )
            )
    elif pred_config.type == "CrossModelPredictor":
        assert (
            pred_config.source_cache_path and pred_config.target_cache_path
        ), "CrossModelPredictor requires source_cache_path and target_cache_path"
        for x in pred_config.x:
            predictors.append(
                CrossModelPredictor(
                    source_cache_path=pred_config.source_cache_path,
                    target_cache_path=pred_config.target_cache_path,
                    k=pred_config.k,
                    n=pred_config.n,
                    x=x,
                )
            )
    else:
        raise ValueError(f"Unknown predictor type: {pred_config.type}")

    return predictors


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run a prediction experiment with the given config."""
    print(f"Running experiment: {config.name}")
    print(f"Model: {config.model}")
    print(f"Dataset: {config.dataset}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(config.model)
    tokenizer = AutoTokenizer.from_pretrained(config.model)

    # Load and process dataset
    ds: DatasetDict = load_dataset(config.dataset)
    tokenized_ds = process_data(ds, tokenizer, split=config.split)

    # Setup output directory
    results_dir = os.path.join(config.output_dir, config.name)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator()

    # Run each predictor
    for pred_config in config.predictors:
        print(f"\n--- Running {pred_config.type} ---")
        predictors = create_predictors(pred_config, model, config.device, results_dir)
        for predictor in predictors:
            print(f"  Running {predictor.name}")
            load_cached(evaluator, results_dir, predictor, tokenized_ds)

            # Clean up hooks for EarlyLayerExit
            if isinstance(predictor, EarlyLayerExit):
                predictor.remove_hook()

    # Print summary
    summary_df = evaluator.summary()
    print("\n=== Summary ===")
    print(summary_df)

    # Save summary
    summary_path = os.path.join(results_dir, "summary.csv")
    summary_df.to_csv(summary_path)
    print(f"\nSummary saved to {summary_path}")

    # Plot pareto curve
    if config.plot_pareto:
        pareto_path = os.path.join(results_dir, "pareto_auroc.png")
        evaluator.plot_pareto(metric="auroc", save_path=pareto_path)

    # Save config for reproducibility
    config_path = os.path.join(results_dir, "config.json")
    config_dict = {
        "name": config.name,
        "model": config.model,
        "dataset": config.dataset,
        "device": config.device,
        "split": config.split,
        "output_dir": config.output_dir,
        "plot_pareto": config.plot_pareto,
        "predictors": [asdict(p) for p in config.predictors],
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    return {"summary": summary_df.to_dict(), "results_dir": results_dir}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction experiments")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    run_experiment(config)
