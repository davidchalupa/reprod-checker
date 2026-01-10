# repro_checker.py
import time
import json
import statistics
import random
import copy
import os
from typing import Callable, Any, Dict, Optional, Tuple, List

import numpy as np
import torch

# Optional nice console output with rich
try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except Exception:
    console = None

def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ReproReport:
    def __init__(self, config):
        self.config = config
        self.runs: List[Dict[str, Any]] = []
        self.env = self._capture_env()

    def add_run(self, run_idx: int, metrics: Dict[str, float], model_state_saved: Optional[str] = None):
        self.runs.append({
            "run_idx": run_idx,
            "metrics": metrics,
            "model_state": model_state_saved,
            "timestamp": time.time(),
        })

    def _capture_env(self):
        return {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

    def summarize(self, metric_tolerance: Dict[str, float] = None):
        if metric_tolerance is None:
            metric_tolerance = {}
        # gather metrics keys
        all_metrics = {}
        for r in self.runs:
            for k, v in r["metrics"].items():
                all_metrics.setdefault(k, []).append(float(v))
        summary = {}
        for k, vals in all_metrics.items():
            mean = statistics.mean(vals)
            stdev = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            mn = min(vals)
            mx = max(vals)
            cv = (stdev / mean) if mean != 0 else (float('inf') if stdev != 0 else 0.0)
            tol = metric_tolerance.get(k, None)
            status = "PASS"
            if tol is not None and stdev > tol:
                status = "FAIL"
            elif cv > 0.01:   # heuristic: >1% CV is a soft warning
                status = "WARN"
            summary[k] = {"mean": mean, "stdev": stdev, "min": mn, "max": mx, "cv": cv, "status": status, "tolerance": tol}
        return summary

    def to_json(self):
        return {
            "config": self.config,
            "env": self.env,
            "runs": self.runs,
            "summary": self.summarize(self.config.get("metric_tolerance", {}))
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    def pretty_print(self):
        summary = self.summarize(self.config.get("metric_tolerance", {}))
        if console:
            table = Table(title="Reproducibility Summary")
            table.add_column("Metric")
            table.add_column("Mean", justify="right")
            table.add_column("Std", justify="right")
            table.add_column("Min", justify="right")
            table.add_column("Max", justify="right")
            table.add_column("CV", justify="right")
            table.add_column("Status")
            for k, v in summary.items():
                table.add_row(k, f"{v['mean']:.6f}", f"{v['stdev']:.6f}", f"{v['min']:.6f}", f"{v['max']:.6f}", f"{v['cv']:.6f}", v['status'])
            console.print(table)
        else:
            print("Reproducibility Summary:")
            for k, v in summary.items():
                print(f" - {k}: mean={v['mean']:.6f}, std={v['stdev']:.6f}, min={v['min']:.6f}, max={v['max']:.6f}, cv={v['cv']:.6f} -> {v['status']}")

class ReprodChecker:
    def __init__(self,
                 train_fn: Callable[[Dict], Any],
                 runs: int = 3,
                 base_seed: int = 42,
                 device: str = "cpu",
                 deterministic: bool = True,
                 metric_keys: Optional[List[str]] = None,
                 metric_tolerance: Optional[Dict[str, float]] = None,
                 save_models: bool = False,
                 out_dir: str = "./repro_out"):
        self.train_fn = train_fn
        self.runs = runs
        self.base_seed = base_seed
        self.device = device
        self.deterministic = deterministic
        self.metric_keys = metric_keys
        self.metric_tolerance = metric_tolerance or {}
        self.save_models = save_models
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.config = {
            "runs": runs, "base_seed": base_seed, "device": device, "deterministic": deterministic,
            "metric_keys": metric_keys, "metric_tolerance": self.metric_tolerance, "save_models": save_models
        }
        self.report = ReproReport(self.config)

    def _apply_determinism(self):
        # Use PyTorch deterministic algorithms where possible
        try:
            torch.use_deterministic_algorithms(self.deterministic)
        except Exception as e:
            if console:
                console.print(f"[yellow]Warning: torch.use_deterministic_algorithms failed: {e}[/yellow]")
            else:
                print("Warning:", e)
        # cudnn flags only relevant if CUDA exists
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = not self.deterministic
            torch.backends.cudnn.deterministic = self.deterministic

    def _parse_return(self, ret) -> Tuple[Dict[str, float], Optional[torch.nn.Module], Optional[str]]:
        model = None
        metrics = None
        model_path = None
        if isinstance(ret, dict):
            metrics = ret
        elif isinstance(ret, tuple) and len(ret) == 2:
            model, metrics = ret
        elif hasattr(ret, "state_dict"):
            model = ret
        else:
            raise RuntimeError("train_fn returned unsupported type. Expected dict or (model, dict) or model with saved metrics externally.")
        if model is not None and self.save_models:
            model_path = os.path.join(self.out_dir, f"model_run_{int(time.time()*1000)}.pt")
            try:
                torch.save(model.state_dict(), model_path)
            except Exception:
                model_path = None
        if metrics is None:
            raise RuntimeError("No metrics returned from train_fn. Please return a dict of metrics (e.g. {'val_loss':..}).")
        return metrics, model, model_path

    def run(self):
        for i in range(self.runs):
            run_seed = self.base_seed + i
            set_global_seeds(run_seed)
            run_cfg = {"seed": run_seed, "device": self.device, "run_idx": i}
            self._apply_determinism()
            if console:
                console.print(f"[green]Starting run {i} with seed {run_seed} on device {self.device}[/green]")
            start = time.time()
            ret = self.train_fn(copy.deepcopy(run_cfg))
            elapsed = time.time() - start
            metrics, model, model_path = self._parse_return(ret)
            metrics["_elapsed_sec"] = elapsed
            self.report.add_run(i, metrics, model_state_saved=model_path)
        return self.report
