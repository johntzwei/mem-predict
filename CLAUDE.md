# Motivation

We will study memorization, with pretraining safety as the overarching research goal. Ensuring that sensitive data is not memorized in pretraining is relevant to copyright, privacy, data poisoning, and dual use concerns. Pretraining safety methods should be used in conjunction with other safety techniques during posttraining.

# Overview

We will use the randomized canaries in Hubble to fit a predictor for LLM memorization based on the model internals. A predictor would help us a) search the training corpus for memorization risks and b) provide better estimates of risk. This is a relaxed membership inference problem, where we assume a white-box setting and access to the training data.

The broader goal is to show that canaries are a promising intervention for AI safety. From the legal view, canaries are proactive, structural interventions that complements many weaker legal arguments at this time. Technically, we may find that canaries are important because they enable us to study the mechanisms of rare behavior, in the model of interest.

## Formulation

We adopt the definition of *k-eidetic memorization* from Carlini et al. (2020), but use the term *k-extractability* as k-extractable sequences can both be memorized or predicted through the model's general ability. In practice, we will mark a training sequence as extractable if given the first k tokens, we can predict the next n tokens (typically k=30, n=20).

The predictor resembles a value function used in RL: it takes a prefix and predicts the probability that the sequence can be extracted.

There is a simple but costly way to compute a sequence's extractability, which is to run a forward pass over the complete sequence and observe whether the model's argmax predictions match the ground truth sequences (to be implemented in `SimpleForward`). Our goal is then to find methods that can predict extractability with less compute, perhaps by observing model internals on only partial sequences, earlier layers, or down projecting model weights.

Methods will be plotted on a pareto curve trading off accuracy and compute costs. The goal is to achieve high accuracy and efficient prediction.

# Background
## Hubble model suite

The Hubble model suite consists of 1B & 8B parameter models trained on 100B & 500B tokens from DCLM corpus. Canary insertions include passages, paraphrases, biographies, chats, and test sets duplicated at 0/1/4/16/64/256 times (<0.01% of training tokens). The models and datasets can be found [here](https://huggingface.co/allegrolab).

Typically, we can use the 1B/100B perturbed model (huggingface identifier: `allegrolab/hubble-1b-100b_toks-perturbed-hf`) to run initial experiments. It is also easiest to work with the randomly inserted Wikpedia passages(huggingface identifier: `allegrolab/passages_wikipedia`) first.

## Technical notes

- Model memorization is thought to occur in earlier layers and be refined by the subsequent layers, in a low rank subspace (see https://arxiv.org/abs/2012.14913 and https://rome.baulab.info/)
- Low-curvature directions correspond to memorization, high-curvature to generalization (see https://arxiv.org/html/2510.24256v2)
- Similar formulations are seen in soft RL (see https://arxiv.org/html/2404.17546v1)