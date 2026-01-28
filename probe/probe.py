from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from datasets.arrow_dataset import Dataset
from torch import nn
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)


class Probe(ABC):
    """Base class for memorization probes."""

    def __init__(self, model: PreTrainedModel, device: str = "cuda") -> None:
        self.model = model
        self.device = device

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

    def save_weights(self, path: str) -> None:
        """Save the classification head weights."""
        # Extract only the score (classification head) parameters and clone to CPU
        score_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items() if "score" in k}
        torch.save(score_state, path)
        print(f"Saved ClassificationHeadProbe weights to {path}")

    def load_weights(self, path: str) -> None:
        """Load the classification head weights."""
        score_state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(score_state, strict=False)
        print(f"Loaded ClassificationHeadProbe weights from {path}")


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, hidden_states: torch.Tensor):
        # only use the last tokens hidden state for classification
        return self.classifier(hidden_states[:, -1, :])


class PretrainedModelWithLinearProbe(nn.Module):
    """Wrapper that combines a frozen pretrained model with a trainable linear probe."""

    def __init__(
            self,
            model: PreTrainedModel,         
            l: int, 
            num_classes: int = 2, 
            pooling: bool = False,
            attn_weighting: bool = False,
        ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.l = l
        self.pooling = pooling
        self.attn_weighting = attn_weighting

        hidden_dim = model.config.hidden_size
        self.classifier = LinearProbe(hidden_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        # Move classifier to the same device as the base model
        model_device = next(model.parameters()).device
        self.classifier.to(model_device)
        self.loss_fn.to(model_device)

        # Expose hf_device_map so Trainer knows not to move the model
        if hasattr(model, "hf_device_map"):
            self.hf_device_map = model.hf_device_map

        # Freeze the base model
        for param in self.model.parameters():
            param.requires_grad = False

    def _extract_hidden_states(self, input_ids, attention_mask=None):
        with torch.no_grad():
            output = self.model(input_ids, output_hidden_states=True, output_attentions=self.attn_weighting, attention_mask=attention_mask)
        return output

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        output = self._extract_hidden_states(input_ids)
        hidden_states = output.hidden_states[self.l + 1]

        if self.attn_weighting:
            attention = output.attentions[self.l] # we dont do l + 1 since model layers include embeddings, attention doesnt
            attention = attention.mean(dim=1) # (B, L, L), average across the attention heads

            attention_diag = attention.diagonal(offset=-1, dim1=-2, dim2=-1)  
            attention_diag = attention_diag.unsqueeze(-1) # B, L-1, 1

            # apply the attention weights now
            hidden_states = hidden_states[:, 1:, :] * attention_diag

            # sum into one feature 
            hidden_states = hidden_states.sum(dim=1, keepdim=True)
        
        if self.pooling:
            hidden_states = torch.mean(hidden_states, dim=-2, keepdim=True)

        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

    @property
    def device(self):
        return next(self.model.parameters()).device


class IntermediateLayerProbe(Probe):
    """Probe that trains a linear classifier on intermediate layer hidden states."""

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
        pooling: bool = False,
        attn_weighting: bool = False,
    ) -> None:
        # Validate layer index before creating wrapper
        num_layers = len(model.model.layers)
        assert l < num_layers, f"Layer {l} invalid, model has {num_layers} layers"

        # Wrap the model with our linear probe
        self.probe_model = PretrainedModelWithLinearProbe(model, l=l, num_classes=2, pooling=pooling, attn_weighting=attn_weighting)
        Probe.__init__(self, self.probe_model, device)

        self.k = k
        self.n = n
        self.l = l
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.pooling = pooling
        self.attn_weighting = attn_weighting

    def fit(self, train_ds: Dataset, labels: np.ndarray) -> None:
        print(f"Training linear probe on layer {self.l} hidden states...")

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
            model=self.probe_model,
            args=training_args,
            train_dataset=train_ds,
        )
        trainer.train()

    def predict_proba(self, ds: Dataset) -> np.ndarray:
        self.probe_model.eval()
        probas = []

        with torch.no_grad():
            for i in tqdm(range(len(ds)), desc="Predicting"):
                input_ids = ds[i]["input_ids"][: self.k + self.n].unsqueeze(0).to(self.probe_model.device)
                outputs = self.probe_model(input_ids)
                prob = torch.softmax(outputs["logits"], dim=-1)[0, 1].cpu().numpy()
                probas.append(prob)

        return np.array(probas)

    @property
    def name(self) -> str:
        return f"IntermediateLayerProbe_k{self.k}_l{self.l}"

    def save_weights(self, path: str) -> None:
        """Save the linear probe classifier weights."""
        state_dict = {k: v.cpu().clone() for k, v in self.probe_model.classifier.state_dict().items()}
        torch.save(state_dict, path)
        print(f"Saved IntermediateLayerProbe weights to {path}")

    def load_weights(self, path: str) -> None:
        """Load the linear probe classifier weights."""
        state_dict = torch.load(path, map_location=self.device)
        self.probe_model.classifier.load_state_dict(state_dict)
        print(f"Loaded IntermediateLayerProbe weights from {path}")
