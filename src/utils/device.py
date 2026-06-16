"""
device.py
---------
GPU / CPU auto-detection and device management.

All training, evaluation, and inference modules call get_device() to
obtain a torch.device — GPU is used when available, CPU otherwise.
This ensures zero code changes are needed between GPU and CPU machines.
"""

import os
import torch
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_device(preferred: str = "auto") -> torch.device:
    """
    Return the best available torch.device.

    Parameters
    ----------
    preferred : str
        'auto'  — use CUDA if available, else CPU  (recommended)
        'cuda'  — force CUDA; raise RuntimeError if unavailable
        'cpu'   — force CPU regardless of GPU availability

    Returns
    -------
    torch.device

    Raises
    ------
    RuntimeError
        If preferred='cuda' but no CUDA-capable GPU is found.
    """
    if preferred == "cpu":
        device = torch.device("cpu")
        logger.info("Device: CPU (forced)")
        return device

    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "preferred='cuda' but torch.cuda.is_available() is False. "
                "Check your CUDA installation or use preferred='auto'."
            )

    # auto or cuda (availability confirmed above)
    if torch.cuda.is_available():
        gpu_id   = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        device   = torch.device(f"cuda:{gpu_id}")
        props    = torch.cuda.get_device_properties(device)
        vram_gb  = props.total_memory / (1024 ** 3)
        logger.info(
            f"Device: GPU — {props.name} | "
            f"VRAM {vram_gb:.1f} GB | "
            f"Compute {props.major}.{props.minor}"
        )
    else:
        device = torch.device("cpu")
        _log_cpu_info()

    return device


def _log_cpu_info() -> None:
    """Log CPU information when falling back to CPU."""
    try:
        import psutil
        cpu_count  = psutil.cpu_count(logical=False)
        thread_cnt = psutil.cpu_count(logical=True)
        ram_gb     = psutil.virtual_memory().total / (1024 ** 3)
        logger.info(
            f"Device: CPU — {cpu_count} cores / {thread_cnt} threads | "
            f"RAM {ram_gb:.1f} GB"
        )
    except ImportError:
        logger.info("Device: CPU (psutil not installed — no hardware details)")


def is_gpu_available() -> bool:
    """Return True if a CUDA-capable GPU is present and accessible."""
    return torch.cuda.is_available()


def move_to_device(obj, device: torch.device):
    """
    Move a tensor, model, or dict of tensors to the target device.

    Parameters
    ----------
    obj : torch.Tensor | torch.nn.Module | dict
    device : torch.device

    Returns
    -------
    Same type as input, on target device.
    """
    if isinstance(obj, (torch.Tensor, torch.nn.Module)):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in obj.items()}
    return obj


def get_device_from_config(training_cfg: dict) -> torch.device:
    """
    Read the 'hardware.device' and 'hardware.gpu_id' fields from
    training.yaml and return the appropriate torch.device.

    Parameters
    ----------
    training_cfg : dict
        Parsed training.yaml config dictionary.

    Returns
    -------
    torch.device
    """
    preferred = training_cfg.get("hardware", {}).get("device", "auto")
    gpu_id    = training_cfg.get("hardware", {}).get("gpu_id", 0)

    if preferred == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
            props  = torch.cuda.get_device_properties(device)
            logger.info(f"Auto-selected GPU: {props.name}")
        else:
            device = torch.device("cpu")
            logger.info("Auto-selected CPU (no CUDA GPU found)")
        return device

    return get_device(preferred)
