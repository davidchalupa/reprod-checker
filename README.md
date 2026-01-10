# Reproducibility Checker — Minimal PoC

Proof-of-concept reproducibility checker for PyTorch training functions.

Quickstart:
1. Create virtualenv: `python -m venv .venv && source .venv/bin/activate`
2. Install deps: `pip install -r requirements.txt`
3. Run demo: `python run_example.py`

This PoC runs a small `train_fn` multiple times with controlled seeds,
collects metrics and prints a reproducibility summary (mean/std/CV).
Designed to work on CPU-only machines.
