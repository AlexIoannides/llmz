"""Tests for utility functions."""

from llmz.utils import LocalFSCheckpointHandler


def test_LocalFSCheckpointHandler():
    assert LocalFSCheckpointHandler is not None
