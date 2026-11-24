"""Logging utilities for training and evaluation."""

import os
from typing import Optional


class SimpleLogger:
    """Console and file metric logger."""

    def __init__(self, log_path: Optional[str] = None) -> None:
        self.log_path = log_path
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    def log(self, message: str) -> None:
        """Print and optionally save to file."""
        print(message)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")
