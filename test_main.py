from typing import Any, Callable

import pytest
import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from main import SimpleForward, SimpleEarlyExit, CrossModelPredictor, PredictionResult, Evaluator, process_data

MODEL_STR = "allegrolab/hubble-1b-100b_toks-perturbed-hf"
DEVICE = "cuda"


@pytest.fixture(scope="module")
def model() -> PreTrainedModel:
    return AutoModelForCausalLM.from_pretrained(MODEL_STR)


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(MODEL_STR)


@pytest.fixture(scope="module")
def dataset() -> DatasetDict:
    return load_dataset('allegrolab/passages_wikipedia')


@pytest.fixture(scope="module")
def tokenized_ds(dataset: DatasetDict, tokenizer: PreTrainedTokenizer) -> Dataset:
    return process_data(dataset, tokenizer, split='train')


def test_simple_forward_memorized_passage(model: PreTrainedModel, tokenized_ds: Dataset) -> None:
    """Last Wikipedia passage is heavily duplicated and should be memorized."""
    predictor = SimpleForward(hf_model=model, device=DEVICE)
    result = predictor.predict(tokenized_ds[-1])

    assert result.ground_truth == True, f"Expected memorized passage to be extractable, got prediction={result.prediction}"


@pytest.fixture
def mock_model(mocker: Any) -> Callable[[torch.Tensor], Any]:
    """Factory fixture for creating mock models with specified logits."""
    def _create(logits: torch.Tensor) -> Any:
        mock = mocker.Mock()
        mock.eval = mocker.Mock()
        mock.to = mocker.Mock(return_value=mock)
        mock_output = mocker.Mock()
        mock_output.logits = logits
        mock.forward = mocker.Mock(return_value=mock_output)
        return mock
    return _create


def test_simple_forward_mocked(mock_model: Callable[[torch.Tensor], Any]) -> None:
    """Test indexing with mocked model: k=3, n=2, tokens [0,1,2,3,4]."""
    k, n, vocab_size = 2, 2, 10
    input_ids = torch.arange(k + n)  # [0,1,2,3]

    # logits[i] predicts token i+1; set correct predictions
    logits = torch.zeros(1, k + n, vocab_size)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 2] = 10.0
    logits[0, 2, 3] = 10.0  # logits[k-1] -> token k
    logits[0, 3, 4] = 10.0  # logits[k-1] -> token k
    # preds = [0, 1, 2, 3]

    model = mock_model(logits)
    predictor = SimpleForward(hf_model=model, device='cpu', k=k, n=n)
    result = predictor.predict({'input_ids': input_ids})

    assert result.predicted_tokens == [2, 3]
    assert result.target_tokens == [2, 3]
    assert result.ground_truth == True


def test_simple_early_exit_equals_simple_forward_at_x_equals_n(model: PreTrainedModel, tokenized_ds: Dataset) -> None:
    """When x=n, SimpleEarlyExit should match SimpleForward on the same example."""
    import tempfile
    import json
    import os
    from dataclasses import asdict

    k, n = 30, 20

    # Run SimpleForward on last example
    sf = SimpleForward(hf_model=model, device=DEVICE, k=k, n=n)
    sf_result = sf.predict(tokenized_ds[-1])

    # Save to temp file for SimpleEarlyExit to load (list format matches Evaluator.save_results)
    data = [asdict(sf_result)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    # Run SimpleEarlyExit with x=n
    see = SimpleEarlyExit(cache_path=temp_path, k=k, n=n, x=n)
    see_result = see._predict(0)

    os.unlink(temp_path)

    assert see_result.prediction == sf_result.prediction
    assert see_result.ground_truth == sf_result.ground_truth
    assert see_result.predicted_tokens == sf_result.predicted_tokens
    assert see_result.target_tokens == sf_result.target_tokens


def test_evaluator_summary() -> None:
    """Test Evaluator.summary with hardcoded predictions and expected metrics."""
    evaluator = Evaluator()

    # 4 predictions: 1 TP, 1 FN, 1 FP, 1 TN
    results = [
        PredictionResult(tokens_generated=20, total_tokens=50,
                         prediction=0.9, ground_truth=True),   # TP
        PredictionResult(tokens_generated=20, total_tokens=50,
                         prediction=0.4, ground_truth=True),   # FN
        PredictionResult(tokens_generated=20, total_tokens=50,
                         prediction=0.6, ground_truth=False),  # FP
        PredictionResult(tokens_generated=20, total_tokens=50,
                         prediction=0.2, ground_truth=False),  # TN
    ]
    evaluator.add_results('test_method', results)

    df = evaluator.summary()

    assert df.loc['test_method', 'accuracy'] == 0.5
    assert df.loc['test_method', 'precision'] == 0.5
    assert df.loc['test_method', 'recall'] == 0.5
    assert df.loc['test_method', 'f1'] == 0.5
    assert df.loc['test_method', 'auroc'] == 0.75


def test_cross_model_equals_early_exit_same_cache(model: PreTrainedModel, tokenized_ds: Dataset) -> None:
    """CrossModelPredictor with same source/target cache should match SimpleEarlyExit."""
    import tempfile
    import json
    import os
    from dataclasses import asdict

    k, n, x = 30, 20, 10

    # Run SimpleForward on a few examples to create cache
    sf = SimpleForward(hf_model=model, device=DEVICE, k=k, n=n)
    sf_results = [sf.predict(tokenized_ds[i]) for i in range(3)]

    # Save cache
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([asdict(r) for r in sf_results], f)
        cache_path = f.name

    # Compare both predictors
    see = SimpleEarlyExit(cache_path=cache_path, k=k, n=n, x=x)
    cmp = CrossModelPredictor(source_cache_path=cache_path, target_cache_path=cache_path, k=k, n=n, x=x)

    for i in range(3):
        see_result = see._predict(i)
        cmp_result = cmp._predict(i)

        assert see_result.prediction == cmp_result.prediction
        assert see_result.ground_truth == cmp_result.ground_truth

    os.unlink(cache_path)
