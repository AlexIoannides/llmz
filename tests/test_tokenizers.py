"""Test for tokenizers."""

from llmz.tokenizers import GPTSmallTextDataset


def test_GPTDataset_encodes_data():
    text = "Attacks ship off the shoulder of Orion."
    dataset = GPTSmallTextDataset(text, 3, 1)
