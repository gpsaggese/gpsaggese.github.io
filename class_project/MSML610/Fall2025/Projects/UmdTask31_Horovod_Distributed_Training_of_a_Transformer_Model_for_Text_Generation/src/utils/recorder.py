"""
Run recording utilities.

Records structured training/validation metrics and run metadata for reproducibility.
"""

import csv
import json
import os
import socket
import subprocess
from datetime import datetime
from typing import Dict, Optional

from .config import save_config


def _safe_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


class RunRecorder:
    """
    Records training/validation metrics and metadata to a run directory (rank 0 only).
    """

    def __init__(
        self,
        base_dir: str,
        rank: int,
        config,
        run_name: Optional[str] = None,
    ):
        self.rank = rank
        self.enabled = (rank == 0)
        self.base_dir = base_dir
        self.config = config

        if not self.enabled:
            self.run_dir = None
            self.metrics_path = None
            self._csv_file = None
            self._csv_writer = None
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = run_name.strip().replace(" ", "_") if run_name else "run"
        self.run_dir = os.path.join(base_dir, f"{ts}_{name}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Prepare CSV
        self.metrics_path = os.path.join(self.run_dir, "metrics.csv")
        self._csv_file = open(self.metrics_path, mode="w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            [
                "phase",  # train/val
                "epoch",
                "update",
                "global_step",
                "lr",
                "loss",
                "perplexity",
                "accuracy",
                "throughput",  # updates/s for train
                "wall_time",
            ]
        )
        self._csv_file.flush()

        # Save config used
        try:
            save_config(config, os.path.join(self.run_dir, "config_used.yaml"))
        except Exception:
            pass

        # Save metadata
        self._write_metadata()

    def _write_metadata(self):
        meta = {
            "timestamp": datetime.now().isoformat(),
            "hostname": socket.gethostname(),
            "git_commit": _safe_git_commit(),
            "slurm": {k: os.environ.get(k) for k in [
                "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_NODELIST", "SLURM_NTASKS",
                "SLURM_NTASKS_PER_NODE", "SLURM_MEM_PER_NODE", "SLURM_GPUS_PER_NODE"
            ]},
        }
        try:
            import torch

            meta.update(
                {
                    "torch_version": getattr(torch, "__version__", "unknown"),
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_version": getattr(torch.version, "cuda", None),
                    "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    "gpu_names": [
                        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                    ]
                    if torch.cuda.is_available()
                    else [],
                }
            )
        except Exception:
            pass

        try:
            import horovod.torch as hvd  # type: ignore

            meta.update(
                {
                    "horovod_version": getattr(hvd, "__version__", "unknown"),
                    "horovod_nccl_built": hvd.nccl_built() if hasattr(hvd, "nccl_built") else None,
                }
            )
        except Exception:
            pass

        with open(os.path.join(self.run_dir, "run_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def log_train_step(
        self,
        *,
        epoch: int,
        update: int,
        global_step: int,
        lr: float,
        loss: float,
        perplexity: float,
        accuracy: Optional[float],
        throughput: Optional[float],
    ):
        if not self.enabled:
            return
        self._csv_writer.writerow(
            [
                "train",
                epoch,
                update,
                global_step,
                lr,
                loss,
                perplexity,
                accuracy if accuracy is not None else "",
                throughput if throughput is not None else "",
                datetime.now().isoformat(),
            ]
        )
        self._csv_file.flush()

    def log_val_epoch(
        self,
        *,
        epoch: int,
        global_step: int,
        loss: float,
        perplexity: float,
        accuracy: Optional[float],
    ):
        if not self.enabled:
            return
        self._csv_writer.writerow(
            [
                "val",
                epoch,
                "",
                global_step,
                "",
                loss,
                perplexity,
                accuracy if accuracy is not None else "",
                "",
                datetime.now().isoformat(),
            ]
        )
        self._csv_file.flush()

    @property
    def directory(self) -> Optional[str]:
        return self.run_dir

    def close(self):
        if self._csv_file:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass

