"""
Device detection utilities
"""
import os

# Enable MPS fallback for unsupported operations (must be before torch import)
if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') is None:
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch


def get_device(preferred_device: str = None):
    """
    Detect and return the best available device
    
    Args:
        preferred_device: Preferred device ('cuda', 'mps', 'cpu'). If None, auto-detects.
    
    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
    """
    if preferred_device:
        # Validate preferred device
        if preferred_device == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        elif preferred_device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        elif preferred_device == 'cpu':
            return 'cpu'
        else:
            print(f"⚠ Warning: Preferred device '{preferred_device}' not available, falling back to auto-detection")
    
    # Auto-detect
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    else:
        return "cpu"

