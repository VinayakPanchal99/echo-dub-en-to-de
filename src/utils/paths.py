"""
Path setup utilities for external dependencies
"""
import os
import sys
from pathlib import Path

# Enable MPS fallback for unsupported operations (needed for torch.istft on Apple Silicon)
if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') is None:
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def setup_paths(workspace_root: Path = None):
    """
    Set up Python paths for external dependencies
    
    Args:
        workspace_root: Root directory of the project. If None, auto-detects.
    
    Returns:
        tuple: (F5_TTS_PATH, SEED_VC_PATH, SEED_VC_AVAILABLE)
    """
    if workspace_root is None:
        # Try to detect workspace root from common locations
        workspace_root = Path(__file__).parent.parent.parent.absolute()
    
    # Add F5-TTS to path
    f5_tts_path = workspace_root / "external" / "F5-TTS" / "src"
    if f5_tts_path.exists():
        sys.path.insert(0, str(f5_tts_path))
    
    # Add Seed-VC to path
    seed_vc_path = workspace_root / "external" / "seed-vc"
    seed_vc_available = False
    if seed_vc_path.exists():
        sys.path.insert(0, str(seed_vc_path))
        try:
            from modules.commons import build_model, recursive_munch, load_checkpoint
            seed_vc_available = True
        except ImportError:
            print("⚠ Warning: Seed-VC modules not found. Please ensure Seed-VC is cloned in external/seed-vc")
    else:
        print("⚠ Warning: Seed-VC path not found. Please clone Seed-VC to external/seed-vc")
    
    return f5_tts_path, seed_vc_path, seed_vc_available

