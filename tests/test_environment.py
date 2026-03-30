import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import EnvironmentChecker, EnvironmentReport


class TestEnvironment:
    def test_check_environment(self):
        checker = EnvironmentChecker()
        report = checker.run_full_check()
        assert isinstance(report, EnvironmentReport)
        check_names = {c.name for c in report.checks}
        assert "Python Version" in check_names
        assert "numpy" in check_names
        assert "opencv-python" in check_names

    def test_get_device(self):
        import torch
        if torch.cuda.is_available():
            assert "cuda" in str(torch.device("cuda"))
        else:
            assert "cpu" in str(torch.device("cpu"))
