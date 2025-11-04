"""
F5-TTS text-to-speech synthesizer module
"""
import gc
import random
import warnings
import time
import os
import sys
from contextlib import contextmanager

import numpy as np
import torch
from f5_tts.api import F5TTS
from scipy import signal

# Suppress F5-TTS/transformers warnings that are not critical for functionality
# These are mostly deprecation warnings from transformers library
warnings.filterwarnings('ignore', message='.*forced_decoder_ids.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*forced_decoder_ids.*')
warnings.filterwarnings('ignore', message='.*past_key_values.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*past_key_values.*')
warnings.filterwarnings('ignore', message='.*attention_mask.*pad token.*eos token.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*attention_mask.*pad token.*eos token.*')
warnings.filterwarnings('ignore', message='.*input name `inputs` is deprecated.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*input name `inputs` is deprecated.*')
warnings.filterwarnings('ignore', message='.*You have passed task=transcribe.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*You have passed task=transcribe.*')
warnings.filterwarnings('ignore', message='.*EncoderDecoderCache.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*EncoderDecoderCache.*')
# Suppress all warnings from transformers
warnings.filterwarnings('ignore', module='transformers.*')
warnings.filterwarnings('ignore', module='.*whisper.*')

@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class F5TTSSynthesizer:
    def __init__(self, config):
        self.config = config
        print(f"\nInitializing F5-TTS...")
        
        # Initialize F5-TTS model
        self.tts_engine = F5TTS(
            model=config.F5_MODEL,
            device=config.DEVICE
        )
        
        # Ensure model is in eval mode for consistent inference
        if hasattr(self.tts_engine, 'ema_model'):
            self.tts_engine.ema_model.eval()
        
        print(f"F5-TTS loaded on {config.DEVICE}")
        print(f"Sample rate: {self.tts_engine.target_sample_rate}Hz")
    
    def _clear_gpu_cache(self, deep=False):
        """
        Clear GPU cache and reset state between segments
        Cross-platform: Works with CUDA (Linux/Windows), MPS (Mac), and CPU
        """
        # CUDA-specific cache clearing (Linux/Windows)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            if deep:
                torch.cuda.ipc_collect()
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
        
        # MPS (Mac) doesn't have explicit cache clearing APIs
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if hasattr(torch.backends.mps, 'synchronize'):
                try:
                    torch.backends.mps.synchronize()
                except:
                    pass
        
        # Force garbage collection
        gc.collect()
        if deep:
            gc.collect()
        
        # Reset model to eval mode
        if hasattr(self.tts_engine, 'ema_model'):
            self.tts_engine.ema_model.eval()
    
    def synthesize_without_voice_cloning(self, text, reference_audio_path, target_duration):
        """
        Generate speech using F5-TTS with minimal voice cloning
        Note: F5-TTS still requires a reference, but we'll use Seed-VC for final voice conversion
        """
        self._clear_gpu_cache()
        
        # F5-TTS requires reference text (transcription of reference audio)
        ref_path_str = str(reference_audio_path)
        
        # Transcribe reference audio (we still need it for F5-TTS, but Seed-VC will do the actual cloning)
        try:
            import logging
            old_log_level = logging.root.level
            
            with suppress_stdout_stderr():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    warnings.filterwarnings("ignore")
                    logging.root.setLevel(logging.ERROR)
                    ref_text = self.tts_engine.transcribe(ref_path_str)
                    logging.root.setLevel(old_log_level)
        except Exception as e:
            ref_text = ""
        
        # Generate seed value for consistency
        if self.config.USE_FIXED_SEED:
            safe_seed = self.config.FIXED_SEED
        else:
            safe_seed = random.randint(0, 4294967295)
        
        # Set PyTorch random seed
        torch.manual_seed(safe_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(safe_seed)
            torch.cuda.manual_seed_all(safe_seed)
        np.random.seed(safe_seed)
        random.seed(safe_seed)
        
        # Generate audio (F5-TTS will use reference minimally)
        # Seed-VC will handle the actual voice cloning later
        # Suppress all verbose output from F5-TTS
        old_tqdm_disable = os.environ.get('TQDM_DISABLE', None)
        os.environ['TQDM_DISABLE'] = '1'
        
        try:
            # Suppress all warnings and output during inference
            import logging
            old_warning_filter = warnings.filters[:]
            old_log_level = logging.root.level
            
            with suppress_stdout_stderr():
                with warnings.catch_warnings():
                    # Catch all warnings
                    warnings.simplefilter("ignore")
                    warnings.filterwarnings("ignore")
                    
                    # Temporarily disable logging
                    logging.root.setLevel(logging.ERROR)
                    
                    wav, sr, spec = self.tts_engine.infer(
                        ref_file=str(reference_audio_path),
                        ref_text=ref_text,
                        gen_text=text,
                        speed=1.0,
                        fix_duration=None,
                        remove_silence=False,
                        seed=safe_seed,
                        nfe_step=self.config.F5_NFE_STEP,
                        cfg_strength=self.config.F5_CFG_STRENGTH,
                        sway_sampling_coef=self.config.F5_SWAY_SAMPLING,
                        target_rms=0.1,
                        cross_fade_duration=0.15
                    )
                    
                    # Restore logging
                    logging.root.setLevel(old_log_level)
        finally:
            if old_tqdm_disable is None:
                os.environ.pop('TQDM_DISABLE', None)
            else:
                os.environ['TQDM_DISABLE'] = old_tqdm_disable
            self._clear_gpu_cache()
        
        return wav, sr
    
    def smooth_audio(self, audio, sample_rate):
        """Apply gentle smoothing to reduce harsh/shrill frequencies"""
        from scipy.signal import butter, filtfilt
        
        nyquist = sample_rate / 2
        cutoff = 4000  # 4kHz cutoff
        order = 4
        
        b, a = butter(order, cutoff / nyquist, btype='low')
        smoothed = filtfilt(b, a, audio)
        
        return smoothed
    
    def adjust_speed(self, audio, target_duration, sample_rate):
        """Adjust audio speed to match target duration"""
        if len(audio) == 0 or target_duration <= 0:
            return audio, 1.0
        
        current_duration = len(audio) / sample_rate
        speed_factor = current_duration / target_duration
        speed_factor_clipped = np.clip(speed_factor, 0.7, 1.5)
        
        target_samples = int(target_duration * sample_rate)
        adjusted_audio = signal.resample(audio, target_samples)
        
        final_duration = len(adjusted_audio) / sample_rate
        actual_speed_factor = current_duration / final_duration if final_duration > 0 else 1.0
        
        return adjusted_audio, actual_speed_factor
    
    def synthesize_all_segments(self, segments, reference_audio):
        """Generate TTS for all segments (without voice cloning - Seed-VC will handle it)"""
        print(f"\n{'='*70}")
        print(f"STEP 4: F5-TTS (Basic TTS - Voice cloning via Seed-VC)")
        print(f"{'='*70}")
        
        results = []
        
        for i, segment in enumerate(segments):
            # Periodic deep cache clear every 10 segments
            if i > 0 and i % 10 == 0:
                self._clear_gpu_cache(deep=True)
            
            try:
                # Generate audio (minimal voice cloning - Seed-VC will handle final conversion)
                audio, sample_rate = self.synthesize_without_voice_cloning(
                    text=segment['text_de'],
                    reference_audio_path=str(reference_audio),
                    target_duration=segment['duration']
                )
                
                # Apply smoothing
                audio = self.smooth_audio(audio, sample_rate)
                
                # Adjust speed for timing
                current_duration = len(audio) / sample_rate
                adjusted_audio, speed_factor = self.adjust_speed(
                    audio,
                    segment['duration'],
                    sample_rate
                )
                
                results.append({
                    'segment_id': i + 1,
                    'audio': adjusted_audio,
                    'text_en': segment['text_en'],
                    'text_de': segment['text_de'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'duration': segment['duration'],
                    'speed_factor': speed_factor,
                    'sample_rate': sample_rate
                })
                
                print(f"Segment {i+1} done")
            except Exception as e:
                print(f"Segment {i+1} ERROR")
                import traceback
                traceback.print_exc()
        
        print(f"\nF5-TTS complete: {len(results)} segments")
        
        # Normalize volume across all segments
        if self.config.NORMALIZE_VOLUME and results:
            results = self.normalize_segment_volumes(results)
        
        return results
    
    def normalize_segment_volumes(self, segments):
        """Normalize volume across all segments for consistent loudness"""
        rms_values = []
        for seg in segments:
            if len(seg['audio']) > 0:
                rms = np.sqrt(np.mean(seg['audio']**2))
                rms_values.append(rms)
        
        if not rms_values:
            return segments
        
        target_rms = np.median(rms_values)
        
        for i, seg in enumerate(segments):
            if len(seg['audio']) > 0:
                current_rms = np.sqrt(np.mean(seg['audio']**2))
                if current_rms > 0:
                    gain = target_rms / current_rms
                    gain = np.clip(gain, 0.5, 2.0)
                    seg['audio'] = seg['audio'] * gain
        
        return segments

# ============================================================================
# STEP 6: VOICE CONVERSION WITH SEED-VC
# ============================================================================

