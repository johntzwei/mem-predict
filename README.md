# mem-predict

Predicting LLM memorization using model internals.

## Usage

### Predict

The `predict` module evaluates different methods for predicting k-extractability of sequences.

```bash
python predict/train.py --config predict/configs/gutenberg_1b.json
```

#### Predict Config

```json
{
  "name": "1b_100b_perturbed",
  "model": "allegrolab/hubble-1b-100b_toks-perturbed-hf",
  "dataset": "allegrolab/passages_gutenberg_popular",
  "device": "cuda",
  "split": "train",
  "output_dir": "results/passages_gutenberg_popular",
  "plot_pareto": true,
  "predictors": [
    {"type": "SimpleForward", "k": 30, "n": 20},
    {"type": "EarlyTokenExit", "k": 30, "n": 20, "x": [1, 5, 10, 15, 20]},
    {"type": "EarlyLayerExit", "k": 30, "n": 20, "l": [12, 13, 14, 15]}
  ]
}
```

**Predictor types:**
- `SimpleForward`: Full forward pass, predicts n tokens given k prefix tokens
- `EarlyTokenExit`: Uses cached SimpleForward results, evaluates on first x tokens
- `EarlyLayerExit`: Exits at layer l and applies lm_head to predict
- `CrossModelPredictor`: Uses a smaller model's predictions to predict larger model extractability

### Probe

The `probe` module trains linear probes on model hidden states to predict extractability.

```bash
python probe/train.py --config probe/configs/wikipedia_1b_classification.json
```

#### Probe Config

```json
{
  "name": "wikipedia_1b_classification_head",
  "model": "allegrolab/hubble-1b-100b_toks-perturbed-hf",
  "dataset": "allegrolab/passages_wikipedia",
  "labels_path": "results/passages_wikipedia/1b_100b_perturbed/SimpleForward_k30_n20.json",
  "device": "cuda",
  "split": "train",
  "n_splits": 3,
  "random_state": 42,
  "output_dir": "results/probe",
  "probe": {
    "type": "ClassificationHeadProbe",
    "k": 30,
    "n": 20,
    "lr": 0.001,
    "epochs": 2
  }
}
```

**Probe types:**
- `ClassificationHeadProbe`: Finetunes a classification head on top of the model
- `IntermediateLayerProbe`: Trains a linear probe on hidden states from layer `l`

**Note:** `labels_path` should point to a SimpleForward results JSON from the predict module.

## Output

Results are saved to the specified `output_dir`:
- `*.json`: Per-predictor/probe results
- `summary.csv`: Metrics summary (accuracy, precision, recall, F1, AUROC)
- `pareto_auroc.png`: Pareto curve of compute vs performance (predict only)
