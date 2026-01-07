import pytest
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from main import SimpleForward, process_data

MODEL_STR = "allegrolab/hubble-1b-100b_toks-perturbed-hf"
DEVICE = "cuda"


@pytest.fixture(scope="module")
def model():
    return AutoModelForCausalLM.from_pretrained(MODEL_STR)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_STR)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset('allegrolab/passages_wikipedia')


@pytest.fixture(scope="module")
def tokenized_ds(dataset, tokenizer):
    return process_data(dataset, tokenizer, split='train')


def test_simple_forward_memorized_passage(model, tokenized_ds):
    """Last Wikipedia passage is heavily duplicated and should be memorized."""
    predictor = SimpleForward(hf_model=model, device=DEVICE)
    result = predictor.predict(tokenized_ds[-1])

    assert result.ground_truth == True, f"Expected memorized passage to be extractable, got prediction={result.prediction}"
