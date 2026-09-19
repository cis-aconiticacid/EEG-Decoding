"""Load the D-096 model under a unique name for whole-repository pytest runs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODEL_PATH = Path(__file__).with_name("model.py")
SPEC = importlib.util.spec_from_file_location("d096_model", MODEL_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load D-096 model from {MODEL_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
