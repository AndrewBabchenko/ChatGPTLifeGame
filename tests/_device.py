"""
Shared device picker for tests - matches training logic
"""
import torch


def pick_device(prefer_gpu: bool = True):
    """
    Matches your training logic:
    - torch_directml if available
    - cuda if available
    - cpu fallback
    """
    if prefer_gpu:
        try:
            import torch_directml
            return torch_directml.device()
        except Exception:
            pass

        if torch.cuda.is_available():
            return torch.device("cuda")

    return torch.device("cpu")


def is_gpu_device(device) -> bool:
    """Check if device is a GPU (DirectML, CUDA, or ROCm)"""
    s = str(device).lower()
    return ("cuda" in s) or ("privateuseone" in s) or ("directml" in s)
