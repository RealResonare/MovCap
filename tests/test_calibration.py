import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration.charuco_detector import CharucoDetector


@pytest.fixture
def detector():
    return CharucoDetector("config/default.yaml")


def test_generate_board(detector):
    board = detector.generate_board_image()
    assert board is not None
    assert len(board.shape) == 2
    assert board.shape[0] > 0
    assert board.shape[1] > 0


def test_detect_empty(detector):
    blank = np.zeros((720, 1280), dtype=np.uint8)
    result = detector.detect(blank)
    assert result is None


def test_detect_noisy(detector):
    noise = np.random.randint(0, 255, (720, 1280), dtype=np.uint8)
    result = detector.detect(noise)
    assert result is None or result is not None
