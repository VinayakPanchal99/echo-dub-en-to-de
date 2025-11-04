#!/usr/bin/env python3
"""
Complete Video Dubbing Pipeline: English to German
Using F5-TTS (without voice cloning) + Seed-VC for voice cloning

Input: MP4 (EN) + SRT (EN)
Output: MP4 (DE audio with Seed-VC voice cloning, original video)
"""

import os
import sys

# Enable MPS fallback for unsupported operations (needed for torch.istft on Apple Silicon)
# This allows MPS to fall back to CPU for operations that aren't yet implemented on MPS
if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') is None:
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import random
import torch
import torchaudio
import soundfile as sf
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import srt
from datetime import timedelta
from scipy import signal
import parselmouth
import subprocess
import json
from pathlib import Path
import gc  # For garbage collection
import yaml

# Add F5-TTS to path
WORKSPACE_ROOT = Path(__file__).parent.parent.absolute()
F5_TTS_PATH = WORKSPACE_ROOT / "external" / "F5-TTS" / "src"
sys.path.insert(0, str(F5_TTS_PATH))

from f5_tts.api import F5TTS

# Add Seed-VC to path (assuming it's cloned in external/seed-vc)
SEED_VC_PATH = WORKSPACE_ROOT / "external" / "seed-vc"
if SEED_VC_PATH.exists():
    sys.path.insert(0, str(SEED_VC_PATH))
    try:
        from modules.commons import build_model, recursive_munch, load_checkpoint
        SEED_VC_AVAILABLE = True
    except ImportError:
        print("⚠ Warning: Seed-VC modules not found. Please ensure Seed-VC is cloned in external/seed-vc")
        SEED_VC_AVAILABLE = False
else:
    print("⚠ Warning: Seed-VC path not found. Please clone Seed-VC to external/seed-vc")
    SEED_VC_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Input/Output
    INPUT_VIDEO = WORKSPACE_ROOT / "inputs" / "Tanzania-2.mp4"
    INPUT_SRT = WORKSPACE_ROOT / "inputs" / "Tanzania-caption.srt"
    OUTPUT_DIR = WORKSPACE_ROOT / "outputs"
    OUTPUT_VIDEO = OUTPUT_DIR / "Tanzania-2_dubbed_de_seedvc.mp4"
    
    # Intermediate files
    TEMP_DIR = WORKSPACE_ROOT / "temp"
    EXTRACTED_AUDIO = TEMP_DIR / "original_audio.wav"
    REFERENCE_AUDIO = TEMP_DIR / "reference_speaker.wav"
    TRANSLATED_SRT = TEMP_DIR / "subtitles_de.srt"
    TTS_AUDIO = TEMP_DIR / "tts_audio.wav"  # Audio before voice conversion
    DUBBED_AUDIO = TEMP_DIR / "dubbed_audio.wav"
    
    # Translation
    NLLB_MODEL = "facebook/nllb-200-distilled-600M"
    SOURCE_LANG = "eng_Latn"  # English
    TARGET_LANG = "deu_Latn"  # German
    
    # F5-TTS Configuration (without voice cloning - just TTS)
    F5_MODEL = "F5TTS_v1_Base"  # Using latest v1 base model
    F5_SAMPLE_RATE = 24000  # F5-TTS native sample rate
    
    # F5-TTS Quality Settings (for basic TTS generation)
    F5_NFE_STEP = 32  # Number of function evaluations
    F5_CFG_STRENGTH = 1.5  # Classifier-free guidance
    F5_SWAY_SAMPLING = -1.0  # Sway sampling coefficient
    
    # Seed-VC Configuration
    SEED_VC_PATH = SEED_VC_PATH
    SEED_VC_CONFIG = SEED_VC_PATH / "configs" / "presets" / "config_dit_mel_seed_uvit_whisper_small_wavenet.yml" if SEED_VC_PATH.exists() else None
    SEED_VC_CHECKPOINT = "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
    SEED_VC_SAMPLE_RATE = 22050  # Seed-VC native sample rate (from config: sr: 22050)
    SEED_VC_DIFFUSION_STEPS = 10  # Number of diffusion steps (more = better quality but slower)
    SEED_VC_CFG_RATE = 0.7  # Classifier-free guidance rate for voice conversion
    
    # Voice Consistency Settings
    USE_FIXED_SEED = True  # Use same seed for all segments (more consistent voice)
    FIXED_SEED = 42  # Seed value (42 is a good default)
    NORMALIZE_VOLUME = True  # Normalize volume across all segments
    CROSS_FADE_DURATION = 0.05  # Cross-fade between segments (seconds) for smoother transitions
    
    # Audio Settings
    FINAL_SAMPLE_RATE = 24000  # Match F5-TTS sample rate initially, then resample if needed
    
    # Processing Settings
    # Cross-platform device detection: CUDA (Linux/Windows), MPS (Mac), or CPU
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = "mps"  # Apple Silicon GPU
    else:
        DEVICE = "cpu"
    SPEED_ADJUSTMENT_RANGE = (0.5, 2.0)
    
    # Prosody Analysis (stored but optional application)
    ENABLE_PROSODY_ANALYSIS = True
    PROSODY_OUTPUT = TEMP_DIR / "prosody_analysis.json"
    
    # Reference Audio Extraction
    # IMPORTANT: Choose a segment with:
    # - Clear speech (no background noise/music)
    # - Natural speaking pace (not too fast/slow)
    # - Good audio quality
    # - 5-10 seconds is optimal for voice cloning
    
    # Intelligent Reference Selection (recommended)
    USE_INTELLIGENT_REFERENCE = True  # Auto-detect best speech segment
    
    # Manual Reference Selection (fallback if auto-detection fails)
    REF_START_TIME = 5.0  # Adjust based on your video
    REF_DURATION = 8.0  # Keep between 5-10 seconds

# ============================================================================
# STEP 1: AUDIO EXTRACTION
# ============================================================================

class AudioExtractor:
    def __init__(self, config):
        self.config = config
    
    def extract_audio_from_video(self, video_path, output_path, sample_rate=24000):
        """Extract full audio track from video"""
        print(f"\n{'='*70}")
        print(f"STEP 1: AUDIO EXTRACTION")
        print(f"{'='*70}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', '1',
            '-y',
            str(output_path)
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✓ Extracted audio: {output_path}")
        
        audio, sr = sf.read(str(output_path))
        duration = len(audio) / sr
        print(f"✓ Duration: {duration:.2f}s, Sample Rate: {sr}Hz")
        
        return output_path, duration
    
    def detect_speech_segments(self, audio, sr, min_silence_duration=0.3, silence_threshold=-40):
        """
        Detect continuous speech segments in audio using energy-based detection
        
        Args:
            audio: Audio signal
            sr: Sample rate
            min_silence_duration: Minimum silence duration to split segments (seconds)
            silence_threshold: Threshold in dB below which is considered silence
        
        Returns:
            List of (start_time, end_time, duration, avg_energy) tuples
        """
        # Calculate RMS energy in small windows
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Compute RMS energy for each frame
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            rms = np.sqrt(np.mean(frame**2))
            energy.append(rms)
        
        energy = np.array(energy)
        
        # Convert to dB
        energy_db = 20 * np.log10(energy + 1e-10)
        
        # Threshold to detect speech
        is_speech = energy_db > silence_threshold
        
        # Find continuous speech segments
        segments = []
        in_speech = False
        start_idx = 0
        
        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                # Start of speech segment
                start_idx = i
                in_speech = True
            elif not speech and in_speech:
                # End of speech segment
                # Check if silence is long enough to split
                silence_duration = 0
                for j in range(i, min(i + int(min_silence_duration / 0.010), len(is_speech))):
                    if j < len(is_speech) and not is_speech[j]:
                        silence_duration += 0.010
                    else:
                        break
                
                if silence_duration >= min_silence_duration:
                    # Long silence - save segment
                    start_time = start_idx * hop_length / sr
                    end_time = i * hop_length / sr
                    duration = end_time - start_time
                    avg_energy = np.mean(energy[start_idx:i])
                    
                    if duration > 0.5:  # Only keep segments > 0.5s
                        segments.append((start_time, end_time, duration, avg_energy))
                    
                    in_speech = False
        
        # Don't forget the last segment if still in speech
        if in_speech:
            start_time = start_idx * hop_length / sr
            end_time = len(is_speech) * hop_length / sr
            duration = end_time - start_time
            avg_energy = np.mean(energy[start_idx:])
            if duration > 0.5:
                segments.append((start_time, end_time, duration, avg_energy))
        
        return segments
    
    def find_best_reference_segment(self, audio_path, target_duration=8.0, min_duration=5.0):
        """
        Intelligently find the best reference audio segment with continuous speech
        
        Args:
            audio_path: Path to audio file
            target_duration: Desired duration (8 seconds - optimal for voice cloning)
            min_duration: Minimum acceptable duration (5 seconds)
        
        Returns:
            (start_time, duration, quality_score)
        """
        print(f"\n📌 Intelligently selecting reference audio...")
        print(f"   Analyzing audio for continuous speech segments...")
        
        audio, sr = sf.read(str(audio_path))
        
        # Detect speech segments
        segments = self.detect_speech_segments(audio, sr)
        
        if not segments:
            print(f"   ⚠ No speech segments detected, using default position")
            return self.config.REF_START_TIME, self.config.REF_DURATION, 0.0
        
        print(f"   Found {len(segments)} speech segments")
        
        # Strategy 1: Find single continuous segment close to target duration
        # Prefer LONGER segments for more stable voice cloning
        best_single = None
        best_score = -1
        
        for start, end, duration, energy in segments:
            if duration >= min_duration:
                # Score based on duration match and energy
                # Prefer segments closer to target duration (8s)
                duration_score = 1.0 - abs(duration - target_duration) / target_duration
                duration_score = max(0, duration_score)
                
                # Bonus for longer segments (more stable voice)
                length_bonus = min(duration / 10.0, 1.0) * 0.2
                
                energy_score = min(energy / 0.1, 1.0)  # Normalize energy
                score = 0.6 * duration_score + 0.2 * energy_score + length_bonus
                
                if score > best_score:
                    best_score = score
                    best_single = (start, min(duration, target_duration), score, energy)
        
        # Strategy 2: Combine consecutive segments if needed
        if not best_single or best_single[1] < min_duration:
            print(f"   No single segment found, trying to combine segments...")
            
            # Try combining consecutive segments
            for i in range(len(segments)):
                combined_duration = 0
                combined_energy = []
                start_time = segments[i][0]
                
                for j in range(i, len(segments)):
                    combined_duration += segments[j][2]
                    combined_energy.append(segments[j][3])
                    
                    if combined_duration >= min_duration:
                        avg_energy = np.mean(combined_energy)
                        duration_to_use = min(combined_duration, target_duration)
                        
                        duration_score = 1.0 - abs(duration_to_use - target_duration) / target_duration
                        duration_score = max(0, duration_score)
                        energy_score = min(avg_energy / 0.1, 1.0)
                        score = 0.7 * duration_score + 0.3 * energy_score
                        
                        if score > best_score:
                            best_score = score
                            best_single = (start_time, duration_to_use, score, avg_energy)
                        
                        break
        
        if best_single:
            start, duration, score, energy = best_single
            print(f"   ✓ Best segment: {start:.2f}s - {start+duration:.2f}s")
            print(f"     Duration: {duration:.2f}s, Energy: {energy:.4f}, Score: {score:.2f}")
            return start, duration, score
        else:
            print(f"   ⚠ No suitable segment found, using default")
            return self.config.REF_START_TIME, self.config.REF_DURATION, 0.0
    
    def extract_reference_audio(self, audio_path, output_path, start_time=None, duration=None):
        """Extract clean reference segment for voice cloning with intelligent selection"""
        
        # Intelligent selection if start_time not provided
        if start_time is None:
            start_time, duration, score = self.find_best_reference_segment(
                audio_path,
                target_duration=8.0,  # Target 8 seconds for optimal voice cloning
                min_duration=5.0      # Minimum 5s, but prefer closer to 8s
            )
        else:
            print(f"\n📌 Extracting reference audio...")
            print(f"   Using manual selection: Start {start_time}s, Duration {duration}s")
        
        audio, sr = sf.read(str(audio_path))
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)
        reference = audio[start_sample:end_sample]
        
        # Verify quality
        rms = np.sqrt(np.mean(reference**2))
        print(f"   Audio quality - RMS: {rms:.4f}, Samples: {len(reference)}")
        
        if rms < 0.01:
            print(f"   ⚠ WARNING: Low audio energy, may affect voice cloning quality")
        
        sf.write(str(output_path), reference, sr)
        print(f"✓ Reference audio saved: {output_path}")
        
        return output_path

# ============================================================================
# STEP 2: SRT PARSING
# ============================================================================

class SRTParser:
    def __init__(self, config):
        self.config = config
    
    def parse_srt(self, srt_path):
        """Parse SRT file into segments"""
        print(f"\n{'='*70}")
        print(f"STEP 2: SRT PARSING")
        print(f"{'='*70}")
        
        with open(str(srt_path), 'r', encoding='utf-8') as f:
            subtitle_generator = srt.parse(f)
            subtitles = list(subtitle_generator)
        
        segments = []
        for sub in subtitles:
            segments.append({
                'index': sub.index,
                'start': sub.start.total_seconds(),
                'end': sub.end.total_seconds(),
                'duration': (sub.end - sub.start).total_seconds(),
                'text_en': sub.content.strip()
            })
        
        print(f"✓ Parsed {len(segments)} segments from SRT")
        print(f"✓ Total duration: {segments[-1]['end']:.2f}s")
        
        return segments

# ============================================================================
# STEP 3: PROSODY ANALYSIS
# ============================================================================

class ProsodyAnalyzer:
    def __init__(self, config):
        self.config = config
    
    def analyze_segment(self, audio_path, start_time, end_time):
        """Analyze prosody features of an audio segment"""
        sound = parselmouth.Sound(str(audio_path))
        segment = sound.extract_part(from_time=start_time, to_time=end_time)
        
        pitch = segment.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        
        intensity = segment.to_intensity()
        intensity_values = intensity.values[0]
        
        duration = end_time - start_time
        
        prosody = {
            'pitch_mean': float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0,
            'pitch_std': float(np.std(pitch_values)) if len(pitch_values) > 0 else 0,
            'pitch_min': float(np.min(pitch_values)) if len(pitch_values) > 0 else 0,
            'pitch_max': float(np.max(pitch_values)) if len(pitch_values) > 0 else 0,
            'intensity_mean': float(np.mean(intensity_values)),
            'intensity_std': float(np.std(intensity_values)),
            'duration': duration
        }
        
        return prosody
    
    def analyze_all_segments(self, audio_path, segments):
        """Analyze prosody for all segments"""
        print(f"\n{'='*70}")
        print(f"STEP 3: PROSODY ANALYSIS")
        print(f"{'='*70}")
        
        if not self.config.ENABLE_PROSODY_ANALYSIS:
            print("⚠ Prosody analysis disabled (skipping)")
            return segments
        
        print(f"Analyzing prosody for {len(segments)} segments...")
        
        for i, segment in enumerate(segments):
            try:
                prosody = self.analyze_segment(
                    audio_path,
                    segment['start'],
                    segment['end']
                )
                segment['prosody'] = prosody
                
                print(f"  [{i+1:2d}] Pitch: {prosody['pitch_mean']:.1f}Hz, "
                      f"Intensity: {prosody['intensity_mean']:.1f}dB")
                
            except Exception as e:
                print(f"  [{i+1:2d}] ⚠ Prosody analysis failed: {e}")
                segment['prosody'] = None
        
        self.config.PROSODY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.config.PROSODY_OUTPUT), 'w') as f:
            json.dump({'segments': [{'index': seg['index'], 'prosody': seg.get('prosody')} for seg in segments]}, f, indent=2)
        
        print(f"✓ Prosody analysis complete")
        print(f"✓ Data saved to: {self.config.PROSODY_OUTPUT}")
        
        return segments

# ============================================================================
# STEP 4: TRANSLATION (NLLB-200)
# ============================================================================

class Translator:
    def __init__(self, config):
        self.config = config
        print(f"\n🔧 Initializing NLLB-200 translator...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.NLLB_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.NLLB_MODEL)
        self.model.to(config.DEVICE)
        self.model.eval()
        
        print(f"✓ NLLB-200 model loaded on {config.DEVICE}")
    
    def translate_text(self, text, source_lang, target_lang):
        """Translate text using NLLB-200"""
        self.tokenizer.src_lang = source_lang
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.config.DEVICE) for k, v in inputs.items()}
        
        # Get target language token ID (compatible with newer transformers)
        if hasattr(self.tokenizer, 'lang_code_to_id'):
            forced_bos_token_id = self.tokenizer.lang_code_to_id[target_lang]
        else:
            # For newer transformers versions
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target_lang)
        
        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
        
        translation = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        
        return translation
    
    def intelligent_pause_strategy(self, text_en, text_de, duration):
        """Adjust translation to preserve timing"""
        en_words = len(text_en.split())
        de_words = len(text_de.split())
        
        en_rate = en_words / duration if duration > 0 else 0
        de_rate = de_words / duration if duration > 0 else 0
        
        rate_ratio = de_rate / en_rate if en_rate > 0 else 1.0
        
        return {
            'text_de': text_de,
            'en_word_count': en_words,
            'de_word_count': de_words,
            'rate_ratio': rate_ratio,
            'needs_adjustment': rate_ratio > 1.3 or rate_ratio < 0.7
        }
    
    def translate_all_segments(self, segments):
        """Translate all segments with intelligent pause strategy"""
        print(f"\n{'='*70}")
        print(f"STEP 4: TRANSLATION (NLLB-200)")
        print(f"{'='*70}")
        
        print(f"Translating {len(segments)} segments from English to German...")
        
        for i, segment in enumerate(segments):
            try:
                text_de = self.translate_text(
                    segment['text_en'],
                    self.config.SOURCE_LANG,
                    self.config.TARGET_LANG
                )
                
                pause_info = self.intelligent_pause_strategy(
                    segment['text_en'],
                    text_de,
                    segment['duration']
                )
                
                segment['text_de'] = pause_info['text_de']
                segment['translation_info'] = pause_info
                
                needs_flag = "⚠" if pause_info['needs_adjustment'] else "✓"
                print(f"  [{i+1:2d}] {needs_flag} EN: '{segment['text_en'][:40]}...'")
                print(f"       DE: '{text_de[:40]}...'")
                
            except Exception as e:
                print(f"  [{i+1:2d}] ✗ Translation failed: {e}")
                segment['text_de'] = segment['text_en']
                segment['translation_info'] = None
        
        self._save_translated_srt(segments)
        
        print(f"\n✓ Translation complete")
        print(f"✓ Translated SRT saved: {self.config.TRANSLATED_SRT}")
        
        return segments
    
    def _save_translated_srt(self, segments):
        """Save translated segments as SRT file"""
        self.config.TRANSLATED_SRT.parent.mkdir(parents=True, exist_ok=True)
        
        srt_content = []
        for seg in segments:
            subtitle = srt.Subtitle(
                index=seg['index'],
                start=timedelta(seconds=seg['start']),
                end=timedelta(seconds=seg['end']),
                content=seg['text_de']
            )
            srt_content.append(subtitle)
        
        with open(str(self.config.TRANSLATED_SRT), 'w', encoding='utf-8') as f:
            f.write(srt.compose(srt_content))

# ============================================================================
# STEP 5: F5-TTS WITHOUT VOICE CLONING (Basic TTS)
# ============================================================================

class F5TTSSynthesizer:
    def __init__(self, config):
        self.config = config
        print(f"\n🔧 Initializing F5-TTS (without voice cloning)...")
        
        # Initialize F5-TTS model
        self.tts_engine = F5TTS(
            model=config.F5_MODEL,
            device=config.DEVICE
        )
        
        # Ensure model is in eval mode for consistent inference
        if hasattr(self.tts_engine, 'ema_model'):
            self.tts_engine.ema_model.eval()
        
        print(f"✓ F5-TTS loaded on {config.DEVICE}")
        print(f"✓ Target sample rate: {self.tts_engine.target_sample_rate}Hz")
        print(f"   Note: Using minimal voice cloning - Seed-VC will handle voice conversion")
    
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
            ref_text = self.tts_engine.transcribe(ref_path_str)
            print(f"   📝 Reference transcription: {ref_text[:50]}...")
        except Exception as e:
            print(f"   ⚠ Could not transcribe reference audio, using empty string: {e}")
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
        try:
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
        finally:
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
        if len(audio) == 0:
            print(f"       ⚠️ WARNING: Empty audio passed to adjust_speed")
            return audio, 1.0
            
        current_duration = len(audio) / sample_rate
        
        if target_duration <= 0:
            print(f"       ⚠️ WARNING: Invalid target duration: {target_duration}")
            return audio, 1.0
        
        speed_factor = current_duration / target_duration
        
        speed_factor_clipped = np.clip(speed_factor, 0.7, 1.5)
        
        if speed_factor != speed_factor_clipped:
            print(f"       ⚠️ Speed factor clipped: {speed_factor:.2f}x → {speed_factor_clipped:.2f}x")
        
        target_samples = int(target_duration * sample_rate)
        adjusted_audio = signal.resample(audio, target_samples)
        
        final_duration = len(adjusted_audio) / sample_rate
        actual_speed_factor = current_duration / final_duration if final_duration > 0 else 1.0
        
        return adjusted_audio, actual_speed_factor
    
    def synthesize_all_segments(self, segments, reference_audio):
        """Generate TTS for all segments (without voice cloning - Seed-VC will handle it)"""
        print(f"\n{'='*70}")
        print(f"STEP 5: F5-TTS (Basic TTS - Voice cloning via Seed-VC)")
        print(f"{'='*70}")
        print(f"   📊 Processing {len(segments)} segments")
        print(f"   🔄 Seed-VC will apply voice cloning in next step")
        
        results = []
        
        for i, segment in enumerate(segments):
            # Periodic deep cache clear every 10 segments
            if i > 0 and i % 10 == 0:
                print(f"  🔄 Deep cache clear at segment {i+1}/{len(segments)}...")
                self._clear_gpu_cache(deep=True)
            
            print(f"  [{i+1:2d}] Generating: '{segment['text_de'][:40]}...' (target: {segment['duration']:.2f}s)")
            
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
                
                # Diagnostic
                final_duration = len(adjusted_audio) / sample_rate
                audio_rms = np.sqrt(np.mean(adjusted_audio**2)) if len(adjusted_audio) > 0 else 0
                
                print(f"       ✓ Generated: {current_duration:.2f}s → Final: {final_duration:.2f}s "
                      f"(speed: {speed_factor:.2f}x, RMS: {audio_rms:.4f})")
                
                if audio_rms < 0.001:
                    print(f"       ⚠️ WARNING: Very quiet audio (RMS: {audio_rms:.4f})")
                
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
                
            except Exception as e:
                print(f"       ✗ TTS failed: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✓ F5-TTS complete: {len(results)} segments")
        
        # Normalize volume across all segments
        if self.config.NORMALIZE_VOLUME and results:
            print(f"\n📊 Normalizing volume across all segments...")
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
        print(f"   Target RMS: {target_rms:.4f}")
        
        for i, seg in enumerate(segments):
            if len(seg['audio']) > 0:
                current_rms = np.sqrt(np.mean(seg['audio']**2))
                if current_rms > 0:
                    gain = target_rms / current_rms
                    gain = np.clip(gain, 0.5, 2.0)
                    seg['audio'] = seg['audio'] * gain
                    
                    new_rms = np.sqrt(np.mean(seg['audio']**2))
                    print(f"   [{seg['segment_id']:2d}] RMS: {current_rms:.4f} → {new_rms:.4f} (gain: {gain:.2f}x)")
        
        print(f"   ✓ Volume normalized across {len(segments)} segments")
        return segments

# ============================================================================
# STEP 6: VOICE CONVERSION WITH SEED-VC
# ============================================================================

class SeedVCVoiceConverter:
    def __init__(self, config):
        self.config = config
        
        if not SEED_VC_AVAILABLE:
            raise ImportError(
                "Seed-VC is not available. Please:\n"
                "1. Clone Seed-VC: git clone https://github.com/Plachtaa/seed-vc.git external/seed-vc\n"
                "2. Install dependencies as per Seed-VC README"
            )
        
        if not config.SEED_VC_CONFIG or not config.SEED_VC_CONFIG.exists():
            raise FileNotFoundError(
                f"Seed-VC config not found at {config.SEED_VC_CONFIG}\n"
                "Please ensure Seed-VC is properly cloned and configured."
            )
        
        print(f"\n🔧 Initializing Seed-VC for voice cloning...")
        
        # Load config
        with open(config.SEED_VC_CONFIG, 'r') as f:
            config_dict = yaml.safe_load(f)
        model_params = recursive_munch(config_dict['model_params'])
        model_params.dit_type = 'DiT'
        
        # Build model
        self.model = build_model(model_params, stage='DiT')
        
        # Load checkpoint
        checkpoint_path = self._get_checkpoint(config.SEED_VC_CHECKPOINT)
        self.model, _, _, _ = load_checkpoint(
            self.model,
            None,
            checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        for key in self.model:
            self.model[key].eval()
            self.model[key].to(config.DEVICE)
        self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        
        # Store model params and config
        self.model_params = model_params
        self.config_dict = config_dict
        self.sr = config_dict['preprocess_params']['sr']
        self.hop_length = config_dict['preprocess_params']['spect_params']['hop_length']
        
        # Load additional modules
        self._load_whisper_model()
        self._load_campplus_model()
        self._load_vocoder()
        self._setup_mel_fn()
        
        print(f"✓ Seed-VC loaded on {config.DEVICE}")
        print(f"   Sample rate: {self.sr}Hz")
    
    def _load_whisper_model(self):
        """Load Whisper model for semantic feature extraction"""
        from transformers import AutoFeatureExtractor, WhisperModel
        
        whisper_name = self.model_params.speech_tokenizer.name if hasattr(self.model_params.speech_tokenizer, 'name') else "openai/whisper-small"
        print(f"  Loading Whisper model: {whisper_name}...")
        
        self.whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(self.config.DEVICE)
        del self.whisper_model.decoder  # Only need encoder
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)
        
        def semantic_fn(waves_16k):
            ori_inputs = self.whisper_feature_extractor(
                [waves_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True
            )
            ori_input_features = self.whisper_model._mask_input_features(
                ori_inputs.input_features, 
                attention_mask=ori_inputs.attention_mask
            ).to(self.config.DEVICE)
            with torch.no_grad():
                ori_outputs = self.whisper_model.encoder(
                    ori_input_features.to(self.whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori
        
        self.semantic_fn = semantic_fn
        print("  ✓ Whisper loaded")
    
    def _load_campplus_model(self):
        """Load CAMPPlus model for style feature extraction"""
        from modules.campplus.DTDNN import CAMPPlus
        from hf_utils import load_custom_model_from_hf
        
        print("  Loading CAMPPlus model...")
        campplus_ckpt_path = load_custom_model_from_hf(
            "funasr/campplus", "campplus_cn_common.bin", config_filename=None
        )
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model.eval()
        self.campplus_model.to(self.config.DEVICE)
        print("  ✓ CAMPPlus loaded")
    
    def _load_vocoder(self):
        """Load BigVGAN vocoder"""
        from modules.bigvgan import bigvgan
        
        print("  Loading BigVGAN vocoder...")
        bigvgan_name = self.model_params.vocoder.name
        self.vocoder = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        self.vocoder.remove_weight_norm()
        self.vocoder = self.vocoder.eval().to(self.config.DEVICE)
        print("  ✓ BigVGAN loaded")
    
    def _setup_mel_fn(self):
        """Setup mel spectrogram function"""
        from modules.audio import mel_spectrogram
        
        mel_fn_args = {
            "n_fft": self.config_dict['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.config_dict['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.config_dict['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.config_dict['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.sr,
            "fmin": 0,
            "fmax": None,
            "center": False
        }
        self.to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)
        
        # Setup overlap parameters
        self.overlap_frame_len = 16
        self.max_context_window = self.sr // self.hop_length * 30
        self.overlap_wave_len = self.overlap_frame_len * self.hop_length
    
    def _get_checkpoint(self, checkpoint_name):
        """Download or locate checkpoint"""
        from huggingface_hub import hf_hub_download
        
        checkpoint_path = os.path.join(
            os.path.expanduser("~/.cache/seed-vc"),
            checkpoint_name
        )
        
        if not os.path.exists(checkpoint_path):
            print(f"  Downloading {checkpoint_name} from HuggingFace...")
            checkpoint_path = hf_hub_download(
                repo_id="Plachta/Seed-VC",
                filename=checkpoint_name,
                cache_dir=os.path.expanduser("~/.cache/seed-vc")
            )
        
        return checkpoint_path
    
    def load_reference_audio(self, reference_path):
        """Load and prepare reference audio for voice conversion"""
        audio, sr = torchaudio.load(str(reference_path))
        
        # Resample if needed
        if sr != self.sr:
            audio = torchaudio.functional.resample(
                audio, sr, self.sr
            )
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        self.reference_audio = audio
        print(f"✓ Reference audio loaded: {reference_path}")
        print(f"   Duration: {audio.shape[1] / self.sr:.2f}s")
    
    def _resample_audio(self, audio, orig_sr, target_sr):
        """Resample audio using torchaudio"""
        if orig_sr == target_sr:
            return audio
        
        # Convert numpy to torch tensor if needed
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
        else:
            audio_tensor = audio
        
        resampled = torchaudio.functional.resample(
            audio_tensor, orig_sr, target_sr
        )
        
        return resampled.squeeze().cpu().numpy() if isinstance(audio, np.ndarray) else resampled
    
    def convert_segment(self, audio, segment_id):
        """Apply voice conversion to one segment using Seed-VC"""
        print(f"  [{segment_id:2d}] VC: Converting voice...", end=" ")
        
        try:
            # Convert audio to torch tensor if needed
            if isinstance(audio, np.ndarray):
                source_audio = torch.from_numpy(audio).float()
            else:
                source_audio = audio.float()
            
            if len(source_audio.shape) == 1:
                source_audio = source_audio.unsqueeze(0)
            
            # Resample to Seed-VC sample rate
            if source_audio.shape[-1] / self.config.F5_SAMPLE_RATE * self.sr != source_audio.shape[-1]:
                source_audio = torchaudio.functional.resample(
                    source_audio, self.config.F5_SAMPLE_RATE, self.sr
                )
            
            source_audio = source_audio.to(self.config.DEVICE)
            ref_audio = self.reference_audio.to(self.config.DEVICE)
            
            # Limit reference audio to 25 seconds
            ref_audio_limited = ref_audio[:, :self.sr * 25]
            
            # Resample to 16kHz for Whisper
            ref_waves_16k = torchaudio.functional.resample(ref_audio_limited, self.sr, 16000)
            converted_waves_16k = torchaudio.functional.resample(source_audio, self.sr, 16000)
            
            # Extract semantic features with Whisper
            if converted_waves_16k.size(-1) <= 16000 * 30:
                S_alt = self.semantic_fn(converted_waves_16k)
            else:
                # Handle long audio with overlapping chunks
                overlapping_time = 5
                S_alt_list = []
                buffer = None
                traversed_time = 0
                while traversed_time < converted_waves_16k.size(-1):
                    if buffer is None:
                        chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
                    else:
                        chunk = torch.cat([
                            buffer, 
                            converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]
                        ], dim=-1)
                    S_alt = self.semantic_fn(chunk)
                    if traversed_time == 0:
                        S_alt_list.append(S_alt)
                    else:
                        S_alt_list.append(S_alt[:, 50 * overlapping_time:])
                    buffer = chunk[:, -16000 * overlapping_time:]
                    traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
                S_alt = torch.cat(S_alt_list, dim=1)
            
            ori_waves_16k = torchaudio.functional.resample(ref_audio_limited, self.sr, 16000)
            S_ori = self.semantic_fn(ori_waves_16k)
            
            # Extract mel spectrograms
            mel = self.to_mel(source_audio.float())
            mel2 = self.to_mel(ref_audio_limited.float())
            
            target_lengths = torch.LongTensor([mel.size(2)]).to(mel.device)
            target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
            
            # Extract style features with CAMPPlus
            feat2 = torchaudio.compliance.kaldi.fbank(
                ref_waves_16k,
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000
            )
            feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
            style2 = self.campplus_model(feat2.unsqueeze(0))
            
            # Length regulation
            cond, _, _, _, _ = self.model.length_regulator(
                S_alt, ylens=target_lengths, n_quantizers=3, f0=None
            )
            prompt_condition, _, _, _, _ = self.model.length_regulator(
                S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
            )
            
            # Voice conversion with chunking
            max_source_window = self.max_context_window - mel2.size(2)
            processed_frames = 0
            generated_wave_chunks = []
            
            with torch.no_grad():
                while processed_frames < cond.size(1):
                    chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
                    is_last_chunk = processed_frames + max_source_window >= cond.size(1)
                    cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
                    
                    device_type_str = self.config.DEVICE if isinstance(self.config.DEVICE, str) else str(self.config.DEVICE)
                    if device_type_str.startswith('mps'):
                        device_type_str = 'cpu'  # MPS doesn't support autocast well
                    with torch.autocast(device_type=device_type_str, dtype=torch.float32):
                        # Voice conversion using CFM inference
                        vc_target = self.model.cfm.inference(
                            cat_condition,
                            torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                            mel2, style2, None,
                            self.config.SEED_VC_DIFFUSION_STEPS,
                    inference_cfg_rate=self.config.SEED_VC_CFG_RATE
                )
                        vc_target = vc_target[:, :, mel2.size(-1):]
                    
                    vc_wave = self.vocoder(vc_target.float())[0]
                    if vc_wave.ndim == 1:
                        vc_wave = vc_wave.unsqueeze(0)
                    
                    if processed_frames == 0:
                        if is_last_chunk:
                            output_wave = vc_wave[0].cpu().numpy()
                            generated_wave_chunks.append(output_wave)
                            break
                        output_wave = vc_wave[0, :-self.overlap_wave_len].cpu().numpy()
                        generated_wave_chunks.append(output_wave)
                        previous_chunk = vc_wave[0, -self.overlap_wave_len:]
                        processed_frames += vc_target.size(2) - self.overlap_frame_len
                    elif is_last_chunk:
                        output_wave = self._crossfade(
                            previous_chunk.cpu().numpy(), 
                            vc_wave[0].cpu().numpy(), 
                            self.overlap_wave_len
                        )
                        generated_wave_chunks.append(output_wave)
                        processed_frames += vc_target.size(2) - self.overlap_frame_len
                    else:
                        output_wave = self._crossfade(
                            previous_chunk.cpu().numpy(),
                            vc_wave[0, :-self.overlap_wave_len].cpu().numpy(),
                            self.overlap_wave_len
                        )
                        generated_wave_chunks.append(output_wave)
                        previous_chunk = vc_wave[0, -self.overlap_wave_len:]
                        processed_frames += vc_target.size(2) - self.overlap_frame_len
            
            converted = np.concatenate(generated_wave_chunks)
            
            # Normalize output
            peak = np.abs(converted).max()
            if peak > 0:
                converted = converted * (0.95 / peak)
            
            print("✓")
            return converted
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: return original audio (resampled if needed)
            if isinstance(audio, np.ndarray):
                return self._resample_audio(
                    audio,
                    self.config.F5_SAMPLE_RATE,
                    self.sr
                )
            return audio
    
    def _crossfade(self, chunk1, chunk2, overlap):
        """Apply crossfade between two audio chunks"""
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
        if len(chunk2) < overlap:
            chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
        else:
            chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
        return chunk2
    
    def convert_all_segments(self, tts_results, reference_audio_path):
        """Apply voice conversion to all segments using Seed-VC"""
        print(f"\n{'='*70}")
        print(f"STEP 6: VOICE CONVERSION (SEED-VC)")
        print(f"{'='*70}")
        
        # Load reference audio
        self.load_reference_audio(reference_audio_path)
        
        print(f"Converting {len(tts_results)} segments with Seed-VC...")
        
        results = []
        for result in tts_results:
            converted_audio = self.convert_segment(
                audio=result['audio'],
                segment_id=result['segment_id']
            )
            
            # Update result with converted audio
            result['audio'] = converted_audio
            result['sample_rate'] = self.config.SEED_VC_SAMPLE_RATE  # Update sample rate
            results.append(result)
        
        print(f"\n✓ Voice conversion complete: {len(results)} segments")
        
        return results

# ============================================================================
# STEP 7: AUDIO ASSEMBLY
# ============================================================================

class AudioAssembler:
    def __init__(self, config):
        self.config = config
    
    def apply_crossfade(self, audio_buffer, new_audio, start_sample, fade_samples):
        """Apply cross-fade between existing audio and new segment"""
        end_sample = min(start_sample + len(new_audio), len(audio_buffer))
        audio_length = end_sample - start_sample
        
        existing_audio = audio_buffer[start_sample:end_sample]
        has_existing = np.any(np.abs(existing_audio) > 1e-6)
        
        if has_existing and fade_samples > 0:
            fade_length = min(fade_samples, audio_length, len(new_audio))
            if fade_length > 0:
                fade_in = np.linspace(0, 1, fade_length)
                fade_out = np.linspace(1, 0, fade_length)
                
                new_audio_slice = new_audio[:audio_length].copy()
                new_audio_slice[:fade_length] = (
                    new_audio_slice[:fade_length] * fade_in +
                    existing_audio[:fade_length] * fade_out
                )
                
                audio_buffer[start_sample:end_sample] = new_audio_slice
            else:
                audio_buffer[start_sample:end_sample] = new_audio[:audio_length]
        else:
            audio_buffer[start_sample:end_sample] = new_audio[:audio_length]
    
    def _resample(self, audio, orig_sr, target_sr):
        """Resample audio"""
        if orig_sr == target_sr:
            return audio
        num_samples = int(len(audio) * target_sr / orig_sr)
        return signal.resample(audio, num_samples)
    
    def assemble_audio(self, segments, total_duration):
        """Assemble all segments with precise timestamp placement"""
        print(f"\n{'='*70}")
        print(f"STEP 7: AUDIO ASSEMBLY")
        print(f"{'='*70}")
        
        print(f"Assembling {len(segments)} segments...")
        if self.config.CROSS_FADE_DURATION > 0:
            print(f"Using {self.config.CROSS_FADE_DURATION}s cross-fade for smooth transitions")
        
        # Use Seed-VC sample rate (segments are converted to this rate)
        sample_rate = self.config.SEED_VC_SAMPLE_RATE
        
        total_samples = int(total_duration * sample_rate)
        final_audio = np.zeros(total_samples, dtype=np.float32)
        
        fade_samples = int(self.config.CROSS_FADE_DURATION * sample_rate)
        
        total_audio_duration = 0
        coverage_samples = 0
        
        for seg in segments:
            start_sample = int(seg['start'] * sample_rate)
            audio = seg['audio']
            
            if len(audio) == 0:
                print(f"  [{seg['segment_id']:2d}] ⚠️ WARNING: Empty audio at {seg['start']:.2f}s")
                continue
            
            # Apply cross-fade if enabled
            if fade_samples > 0:
                self.apply_crossfade(final_audio, audio, start_sample, fade_samples)
            else:
                end_sample = min(start_sample + len(audio), total_samples)
                audio_slice = audio[:end_sample - start_sample]
                final_audio[start_sample:end_sample] = audio_slice
            
            actual_duration = len(audio) / sample_rate
            total_audio_duration += actual_duration
            coverage_samples += len(audio)
            
            print(f"  [{seg['segment_id']:2d}] Placed at {seg['start']:.2f}s "
                  f"(duration: {actual_duration:.2f}s, samples: {len(audio)})")
        
        # Upsample to final quality if needed
        if sample_rate != self.config.FINAL_SAMPLE_RATE:
            print(f"\n📊 Upsampling from {sample_rate}Hz to {self.config.FINAL_SAMPLE_RATE}Hz...")
            final_audio = self._resample(final_audio, sample_rate, self.config.FINAL_SAMPLE_RATE)
            sample_rate = self.config.FINAL_SAMPLE_RATE
        
        # Calculate coverage statistics
        coverage_percent = (coverage_samples / total_samples) * 100 if total_samples > 0 else 0
        
        print(f"\n📊 Assembly Statistics:")
        print(f"   Total duration: {total_duration:.2f}s")
        print(f"   Audio coverage: {total_audio_duration:.2f}s")
        print(f"   Coverage: {coverage_percent:.1f}%")
        print(f"   Silence: {100 - coverage_percent:.1f}%")
        
        if coverage_percent < 50:
            print(f"   ⚠️ WARNING: Less than 50% audio coverage - video will have long silences!")
        
        print(f"\n✓ Audio assembled: {total_duration:.2f}s")
        
        return final_audio, sample_rate
    
    def save_audio(self, audio, output_path, sample_rate):
        """Save assembled audio"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, sample_rate)
        print(f"✓ Saved dubbed audio: {output_path}")

# ============================================================================
# STEP 8: VIDEO MERGE
# ============================================================================

class VideoMerger:
    def __init__(self, config):
        self.config = config
    
    def merge_audio_video(self, video_path, audio_path, output_path):
        """Replace video audio track with dubbed audio"""
        print(f"\n{'='*70}")
        print(f"STEP 8: VIDEO MERGE")
        print(f"{'='*70}")
        
        print(f"Merging audio with video...")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-i', str(audio_path),
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            '-y',
            str(output_path)
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print(f"✓ Final video saved: {output_path}")

# ============================================================================
# STEP 9: EVALUATION & REPORT GENERATION
# ============================================================================

class QualityEvaluator:
    def __init__(self, config):
        self.config = config
        self.speaker_encoder = None
        
    def _load_speaker_encoder(self):
        """Load speaker verification model for similarity computation"""
        if self.speaker_encoder is None:
            try:
                import warnings
                warnings.filterwarnings('ignore', category=UserWarning)
                
                from speechbrain.pretrained import EncoderClassifier
                print(f"\n🔧 Loading speaker verification model...")
                self.speaker_encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="temp/speaker_encoder"
                )
                print(f"✓ Speaker encoder loaded")
            except ImportError:
                print(f"⚠ SpeechBrain not installed")
                print(f"   Install with: pip install speechbrain")
                self.speaker_encoder = None
            except Exception as e:
                print(f"⚠ Could not load speaker encoder: {e}")
                self.speaker_encoder = None
    
    def compute_speaker_similarity(self, original_audio_path, dubbed_audio_path):
        """Compute cosine similarity between original and dubbed audio"""
        try:
            self._load_speaker_encoder()
            
            if self.speaker_encoder is None:
                return None, None, "Speaker encoder not available"
            
            original, sr1 = sf.read(str(original_audio_path))
            dubbed, sr2 = sf.read(str(dubbed_audio_path))
            
            emb_original = self.speaker_encoder.encode_batch(
                torch.FloatTensor(original).unsqueeze(0)
            )
            emb_dubbed = self.speaker_encoder.encode_batch(
                torch.FloatTensor(dubbed).unsqueeze(0)
            )
            
            similarity = torch.nn.functional.cosine_similarity(
                emb_original, emb_dubbed
            ).item()
            
            return similarity, similarity, None
            
        except Exception as e:
            return None, None, str(e)
    
    def compute_timing_alignment(self, original_audio_path, dubbed_audio_path):
        """Compare durations of original and dubbed audio"""
        try:
            original, sr1 = sf.read(str(original_audio_path))
            dubbed, sr2 = sf.read(str(dubbed_audio_path))
            
            original_duration = len(original) / sr1
            dubbed_duration = len(dubbed) / sr2
            
            drift = dubbed_duration - original_duration
            drift_percent = (drift / original_duration) * 100 if original_duration > 0 else 0
            
            return original_duration, dubbed_duration, drift, drift_percent
            
        except Exception as e:
            print(f"⚠ Could not compute timing alignment: {e}")
            return None, None, None, None
    
    def compute_audio_quality_metrics(self, audio_path):
        """Compute basic audio quality metrics"""
        try:
            audio, sr = sf.read(str(audio_path))
            
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))
            dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
            
            return {
                'rms': rms,
                'peak': peak,
                'dynamic_range_db': dynamic_range,
                'sample_rate': sr,
                'duration': len(audio) / sr
            }
            
        except Exception as e:
            print(f"⚠ Could not compute audio quality: {e}")
            return None
    
    def generate_evaluation_report(self, segments=None):
        """Generate comprehensive evaluation report"""
        print(f"\n{'='*70}")
        print(f"STEP 9: QUALITY EVALUATION & REPORT GENERATION")
        print(f"{'='*70}")
        
        report_path = self.config.OUTPUT_DIR / "evaluation_report_seedvc.md"
        
        # Compute metrics
        print("Computing speaker similarity...")
        full_similarity, segment_mean, error = self.compute_speaker_similarity(
            self.config.EXTRACTED_AUDIO,
            self.config.DUBBED_AUDIO
        )
        
        print("Computing timing alignment...")
        orig_dur, dubbed_dur, drift, drift_pct = self.compute_timing_alignment(
            self.config.EXTRACTED_AUDIO,
            self.config.DUBBED_AUDIO
        )
        
        print("Computing audio quality metrics...")
        orig_quality = self.compute_audio_quality_metrics(self.config.EXTRACTED_AUDIO)
        dubbed_quality = self.compute_audio_quality_metrics(self.config.DUBBED_AUDIO)
        
        # Generate markdown report
        with open(str(report_path), 'w', encoding='utf-8') as f:
            f.write("# Video Translation Evaluation Report (Seed-VC)\n\n")
            f.write(f"**Generated**: {timedelta(seconds=0)}\n\n")
            f.write(f"**Source**: {self.config.INPUT_VIDEO.name}\n")
            f.write(f"**Target Language**: German\n\n")
            
            # Translation Quality
            f.write("## Translation Quality\n\n")
            if segments:
                f.write(f"- **Total Segments**: {len(segments)}\n")
                adjusted_count = sum(1 for seg in segments if seg.get('translation_info', {}).get('needs_adjustment', False))
                f.write(f"- **Segments Needing Timing Adjustment**: {adjusted_count}/{len(segments)}\n")
                
                speed_factors = [seg.get('speed_factor', 1.0) for seg in segments if 'speed_factor' in seg]
                if speed_factors:
                    avg_speed = np.mean(speed_factors)
                    f.write(f"- **Average Speed Factor**: {avg_speed:.2f}x\n")
            f.write("\n")
            
            # Speaker Similarity
            f.write("## Speaker Similarity\n\n")
            if full_similarity is not None:
                f.write(f"- **Full Audio Cosine Similarity**: {full_similarity:.4f}\n")
                f.write(f"- **Threshold (≥0.65)**: {'✅ PASS' if full_similarity >= 0.65 else '❌ FAIL'}\n")
                if segment_mean is not None:
                    f.write(f"- **Segment Mean**: {segment_mean:.4f}\n")
            else:
                f.write(f"- **Status**: ⚠️ Not computed\n")
                if error:
                    f.write(f"- **Reason**: {error}\n")
            f.write("\n")
            
            # Timing Alignment
            f.write("## Timing Alignment\n\n")
            if orig_dur is not None:
                f.write(f"- **Original Duration**: {orig_dur:.2f}s\n")
                f.write(f"- **Synthesized Duration**: {dubbed_dur:.2f}s\n")
                f.write(f"- **Drift**: {drift:+.2f}s ({drift_pct:+.1f}%)\n")
                f.write(f"- **Threshold (±5%)**: {'✅ PASS' if abs(drift_pct) <= 5 else '❌ FAIL'}\n")
            else:
                f.write(f"- **Status**: ⚠️ Not computed\n")
            f.write("\n")
            
            # Audio Quality Metrics
            f.write("## Audio Quality Metrics\n\n")
            if orig_quality and dubbed_quality:
                f.write("### Original Audio\n")
                f.write(f"- **Sample Rate**: {orig_quality['sample_rate']}Hz\n")
                f.write(f"- **RMS Level**: {orig_quality['rms']:.4f}\n")
                f.write(f"- **Peak Amplitude**: {orig_quality['peak']:.4f}\n")
                f.write(f"- **Dynamic Range**: {orig_quality['dynamic_range_db']:.2f}dB\n\n")
                
                f.write("### Dubbed Audio\n")
                f.write(f"- **Sample Rate**: {dubbed_quality['sample_rate']}Hz\n")
                f.write(f"- **RMS Level**: {dubbed_quality['rms']:.4f}\n")
                f.write(f"- **Peak Amplitude**: {dubbed_quality['peak']:.4f}\n")
                f.write(f"- **Dynamic Range**: {dubbed_quality['dynamic_range_db']:.2f}dB\n")
            f.write("\n")
            
            # Model Configuration
            f.write("## Configuration\n\n")
            f.write(f"- **TTS Model**: {self.config.F5_MODEL} (without voice cloning)\n")
            f.write(f"- **Voice Conversion**: Seed-VC\n")
            f.write(f"- **F5-TTS Sample Rate**: {self.config.F5_SAMPLE_RATE}Hz\n")
            f.write(f"- **Seed-VC Sample Rate**: {self.config.SEED_VC_SAMPLE_RATE}Hz\n")
            f.write(f"- **Final Sample Rate**: {self.config.FINAL_SAMPLE_RATE}Hz\n")
            f.write(f"- **Seed-VC Diffusion Steps**: {self.config.SEED_VC_DIFFUSION_STEPS}\n")
            f.write(f"- **Seed-VC CFG Rate**: {self.config.SEED_VC_CFG_RATE}\n")
            f.write(f"- **Reference Duration**: {self.config.REF_DURATION}s (starting at {self.config.REF_START_TIME}s)\n")
            f.write("\n")
            
            f.write("## Voice Consistency Settings\n\n")
            f.write(f"- **Fixed Seed**: {'✅ Enabled' if self.config.USE_FIXED_SEED else '❌ Disabled'}")
            if self.config.USE_FIXED_SEED:
                f.write(f" (seed={self.config.FIXED_SEED})")
            f.write("\n")
            f.write(f"- **Volume Normalization**: {'✅ Enabled' if self.config.NORMALIZE_VOLUME else '❌ Disabled'}\n")
            f.write(f"- **Cross-fade**: {self.config.CROSS_FADE_DURATION}s between segments\n")
            f.write("\n")
        
        print(f"✓ Evaluation report saved: {report_path}")
        return report_path

# ============================================================================
# MAIN PIPELINE
# ============================================================================

class VideoDubbingPipeline:
    def __init__(self, config):
        self.config = config
        
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        self.audio_extractor = AudioExtractor(config)
        self.srt_parser = SRTParser(config)
        self.prosody_analyzer = ProsodyAnalyzer(config)
        self.translator = Translator(config)
        self.tts_generator = F5TTSSynthesizer(config)
        self.voice_converter = SeedVCVoiceConverter(config) if SEED_VC_AVAILABLE else None
        self.audio_assembler = AudioAssembler(config)
        self.video_merger = VideoMerger(config)
        self.evaluator = QualityEvaluator(config)
    
    def run(self):
        """Execute complete pipeline"""
        print("\n" + "="*70)
        print("🎬 VIDEO DUBBING PIPELINE: F5-TTS + Seed-VC")
        print("   English → German with Seed-VC Voice Cloning")
        print("="*70)
        
        # Step 1: Extract audio
        audio_path, duration = self.audio_extractor.extract_audio_from_video(
            self.config.INPUT_VIDEO,
            self.config.EXTRACTED_AUDIO
        )
        
        # Extract reference audio (intelligent or manual)
        if self.config.USE_INTELLIGENT_REFERENCE:
            reference_path = self.audio_extractor.extract_reference_audio(
                audio_path,
                self.config.REFERENCE_AUDIO
            )
        else:
            reference_path = self.audio_extractor.extract_reference_audio(
                audio_path,
                self.config.REFERENCE_AUDIO,
                self.config.REF_START_TIME,
                self.config.REF_DURATION
            )
        
        # Step 2: Parse SRT
        segments = self.srt_parser.parse_srt(self.config.INPUT_SRT)
        
        # Step 3: Prosody analysis
        segments = self.prosody_analyzer.analyze_all_segments(audio_path, segments)
        
        # Step 4: Translation
        segments = self.translator.translate_all_segments(segments)
        
        # Step 5: TTS without voice cloning (basic generation)
        tts_results = self.tts_generator.synthesize_all_segments(segments, reference_path)
        
        # Step 6: Voice conversion with Seed-VC
        if self.voice_converter is None:
            raise RuntimeError("Seed-VC voice converter is not available. Please check Seed-VC installation.")
        
        vc_results = self.voice_converter.convert_all_segments(tts_results, reference_path)
        
        # Step 7: Audio assembly
        final_audio, sample_rate = self.audio_assembler.assemble_audio(vc_results, duration)
        self.audio_assembler.save_audio(final_audio, self.config.DUBBED_AUDIO, sample_rate)
        
        # Step 8: Video merge
        self.video_merger.merge_audio_video(
            self.config.INPUT_VIDEO,
            self.config.DUBBED_AUDIO,
            self.config.OUTPUT_VIDEO
        )
        
        # Step 9: Quality evaluation and report generation
        evaluation_report = self.evaluator.generate_evaluation_report(segments=vc_results)
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE")
        print("="*70)
        print(f"📁 Output Video: {self.config.OUTPUT_VIDEO}")
        print(f"📊 Evaluation Report: {evaluation_report}")
        print(f"📊 Prosody Data: {self.config.PROSODY_OUTPUT}")
        print(f"📝 Translated SRT: {self.config.TRANSLATED_SRT}")
        print("="*70 + "\n")

# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    config = Config()
    pipeline = VideoDubbingPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()
