"""
Configuration class for the video dubbing pipeline
"""
import os
from pathlib import Path

# Enable MPS fallback for unsupported operations (must be before any torch imports)
if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') is None:
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from src.utils.device import get_device
from src.utils.paths import setup_paths


class Config:
    """Configuration for the video dubbing pipeline"""
    
    def __init__(self, workspace_root: Path = None):
        """
        Initialize configuration
        
        Args:
            workspace_root: Root directory of the project. If None, auto-detects.
        """
        # Setup paths and get workspace root
        if workspace_root is None:
            workspace_root = Path(__file__).parent.parent.parent.absolute()
        
        self.WORKSPACE_ROOT = workspace_root
        self.F5_TTS_PATH, self.SEED_VC_PATH, self.SEED_VC_AVAILABLE = setup_paths(workspace_root)
        
        # Input/Output (defaults - can be overridden)
        self.INPUT_VIDEO = workspace_root / "inputs" / "Tanzania-2.mp4"
        self.INPUT_SRT = workspace_root / "inputs" / "Tanzania-caption.srt"
        self.OUTPUT_DIR = workspace_root / "outputs"
        self.OUTPUT_VIDEO = self.OUTPUT_DIR / "Tanzania-2_dubbed_de_seedvc.mp4"
        
        # Intermediate files
        self.TEMP_DIR = workspace_root / "temp"
        self.EXTRACTED_AUDIO = self.TEMP_DIR / "original_audio.wav"
        self.REFERENCE_AUDIO = self.TEMP_DIR / "reference_speaker.wav"
        self.TRANSLATED_SRT = self.TEMP_DIR / "subtitles_de.srt"
        self.TTS_AUDIO = self.TEMP_DIR / "tts_audio.wav"  # Audio before voice conversion (optional, for debugging)
        self.DUBBED_AUDIO = self.TEMP_DIR / "dubbed_audio.wav"
        
        # Translation
        self.NLLB_MODEL = "facebook/nllb-200-distilled-600M"
        self.SOURCE_LANG = "eng_Latn"  # English
        self.TARGET_LANG = "deu_Latn"  # German
        
        # F5-TTS Configuration (without voice cloning - just TTS)
        self.F5_MODEL = "F5TTS_v1_Base"  # Using latest v1 base model
        self.F5_SAMPLE_RATE = 24000  # F5-TTS native sample rate
        
        # F5-TTS Quality Settings (for basic TTS generation)
        self.F5_NFE_STEP = 32  # Number of function evaluations
        self.F5_CFG_STRENGTH = 1.5  # Classifier-free guidance
        self.F5_SWAY_SAMPLING = -1.0  # Sway sampling coefficient
        
        # Seed-VC Configuration
        self.SEED_VC_CONFIG = (
            self.SEED_VC_PATH / "configs" / "presets" / "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
            if self.SEED_VC_PATH.exists() else None
        )
        self.SEED_VC_CHECKPOINT = "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
        self.SEED_VC_SAMPLE_RATE = 22050  # Seed-VC native sample rate (from config: sr: 22050)
        self.SEED_VC_DIFFUSION_STEPS = 10  # Number of diffusion steps (more = better quality but slower)
        self.SEED_VC_CFG_RATE = 0.7  # Classifier-free guidance rate for voice conversion
        
        # Voice Consistency Settings
        self.USE_FIXED_SEED = True  # Use same seed for all segments (more consistent voice)
        self.FIXED_SEED = 42  # Seed value (42 is a good default)
        self.NORMALIZE_VOLUME = True  # Normalize volume across all segments
        self.CROSS_FADE_DURATION = 0.05  # Cross-fade between segments (seconds) for smoother transitions
        
        # Audio Settings
        # Final output sample rate - keep at Seed-VC native rate to avoid unnecessary upsampling
        # Pipeline: F5-TTS (24000Hz) -> Seed-VC (22050Hz) -> Keep at 22050Hz (FFmpeg supports this)
        self.FINAL_SAMPLE_RATE = 22050  # Match Seed-VC output rate (no upsampling needed)
        
        # Processing Settings
        # Check environment variable first (useful for Docker)
        preferred_device = os.environ.get('DEVICE')
        self.DEVICE = get_device(preferred_device)  # Auto-detect device or use DEVICE env var
        self.SPEED_ADJUSTMENT_RANGE = (0.7, 1.5)  # Speed adjustment range for timing alignment
        self.VOLUME_GAIN_RANGE = (0.5, 2.0)  # Volume gain range for normalization
        
        # Multi-threading Settings (can be overridden via environment variables)
        self.ENABLE_MULTITHREADING = os.environ.get('ENABLE_MULTITHREADING', 'true').lower() == 'true'
        self.MAX_WORKERS_TRANSLATION = int(os.environ.get('MAX_WORKERS_TRANSLATION', '4'))
        self.MAX_WORKERS_TTS = int(os.environ.get('MAX_WORKERS_TTS', '2'))
        self.MAX_WORKERS_VC = int(os.environ.get('MAX_WORKERS_VC', '2'))
        
        # Prosody Analysis (stored but optional application)
        self.ENABLE_PROSODY_ANALYSIS = True
        self.PROSODY_OUTPUT = self.TEMP_DIR / "prosody_analysis.json"
        
        # Reference Audio Extraction
        # IMPORTANT: Choose a segment with:
        # - Clear speech (no background noise/music)
        # - Natural speaking pace (not too fast/slow)
        # - Good audio quality
        # - 5-10 seconds is optimal for voice cloning
        
        # Intelligent Reference Selection (recommended)
        self.USE_INTELLIGENT_REFERENCE = True  # Auto-detect best speech segment
        
        # Manual Reference Selection (fallback if auto-detection fails)
        self.REF_START_TIME = 5.0  # Adjust based on your video
        self.REF_DURATION = 8.0  # Keep between 5-10 seconds

