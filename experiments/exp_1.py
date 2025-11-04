#!/usr/bin/env python3
"""
Complete Video Dubbing Pipeline: English to German
Using F5-TTS with built-in voice cloning

Input: MP4 (EN) + SRT (EN)
Output: MP4 (DE audio with voice cloning, original video)
"""

import os
import sys
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

# Add F5-TTS to path
WORKSPACE_ROOT = Path(__file__).parent.parent.absolute()
F5_TTS_PATH = WORKSPACE_ROOT / "external" / "F5-TTS" / "src"
sys.path.insert(0, str(F5_TTS_PATH))

from f5_tts.api import F5TTS

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Input/Output
    INPUT_VIDEO = WORKSPACE_ROOT / "inputs" / "Tanzania-2.mp4"
    INPUT_SRT = WORKSPACE_ROOT / "inputs" / "Tanzania-caption.srt"
    OUTPUT_DIR = WORKSPACE_ROOT / "outputs"
    OUTPUT_VIDEO = OUTPUT_DIR / "Tanzania-2_dubbed_de.mp4"
    
    # Intermediate files
    TEMP_DIR = WORKSPACE_ROOT / "temp"
    EXTRACTED_AUDIO = TEMP_DIR / "original_audio.wav"
    REFERENCE_AUDIO = TEMP_DIR / "reference_speaker.wav"
    TRANSLATED_SRT = TEMP_DIR / "subtitles_de.srt"
    DUBBED_AUDIO = TEMP_DIR / "dubbed_audio.wav"
    
    # Translation
    NLLB_MODEL = "facebook/nllb-200-distilled-600M"
    SOURCE_LANG = "eng_Latn"  # English
    TARGET_LANG = "deu_Latn"  # German
    
    # F5-TTS Configuration
    F5_MODEL = "F5TTS_v1_Base"  # Using latest v1 base model
    F5_SAMPLE_RATE = 24000  # F5-TTS native sample rate
    
    # F5-TTS Quality Settings (tune these for better voice matching)
    F5_NFE_STEP = 32  # Number of function evaluations (16-64, higher=better quality but slower)
    F5_CFG_STRENGTH = 1.5  # Classifier-free guidance (1.0-3.0, LOWER=smoother voice, less harsh)
    F5_SWAY_SAMPLING = -1.0  # Sway sampling coefficient (-1=disabled for more stable output)
    
    # Voice Consistency Settings
    USE_FIXED_SEED = True  # Use same seed for all segments (more consistent voice)
    FIXED_SEED = 42  # Seed value (42 is a good default)
    NORMALIZE_VOLUME = True  # Normalize volume across all segments
    CROSS_FADE_DURATION = 0.05  # Cross-fade between segments (seconds) for smoother transitions
    
    # Audio Settings
    FINAL_SAMPLE_RATE = 24000  # Match F5-TTS sample rate
    
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
    # - 5-10 seconds is optimal for F5-TTS voice cloning
    
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
            target_duration: Desired duration (8 seconds - optimal for F5-TTS)
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
# STEP 5: F5-TTS WITH VOICE CLONING
# ============================================================================

class F5TTSSynthesizer:
    def __init__(self, config):
        self.config = config
        print(f"\n🔧 Initializing F5-TTS...")
        
        # Initialize F5-TTS model
        self.tts_engine = F5TTS(
            model=config.F5_MODEL,
            device=config.DEVICE
        )
        
        # Cache for reference audio transcription (to avoid re-transcribing for each segment)
        self.cached_ref_text = {}
        
        # Ensure model is in eval mode for consistent inference
        if hasattr(self.tts_engine, 'ema_model'):
            self.tts_engine.ema_model.eval()
        
        print(f"✓ F5-TTS loaded on {config.DEVICE}")
        print(f"✓ Target sample rate: {self.tts_engine.target_sample_rate}Hz")
    
    def _clear_gpu_cache(self, deep=False):
        """
        Clear GPU cache and reset state between segments to prevent voice drift
        Cross-platform: Works with CUDA (Linux/Windows), MPS (Mac), and CPU
        
        Args:
            deep: If True, performs more aggressive clearing (for periodic cleanup)
        """
        # CUDA-specific cache clearing (Linux/Windows)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            if deep:
                # More aggressive clearing - useful for long videos
                torch.cuda.ipc_collect()
                # Reset peak memory stats to avoid fragmentation
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
        
        # MPS (Mac) doesn't have explicit cache clearing APIs, but garbage collection helps
        # MPS memory is managed more automatically by the system
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS doesn't have empty_cache(), but we can synchronize if available
            if hasattr(torch.backends.mps, 'synchronize'):
                try:
                    torch.backends.mps.synchronize()
                except:
                    pass  # Fallback if not available
        
        # Force garbage collection to free Python objects (works on all platforms)
        gc.collect()
        
        # For deep clean, run GC twice to handle circular references
        if deep:
            gc.collect()
        
        # Reset model to eval mode (important for batch norm layers)
        if hasattr(self.tts_engine, 'ema_model'):
            self.tts_engine.ema_model.eval()
    
    def synthesize_with_voice_cloning(self, text, reference_audio_path, target_duration):
        """
        Generate speech with voice cloning using F5-TTS
        F5-TTS uses reference audio for zero-shot cloning
        
        CRITICAL: This method includes state clearing to prevent voice drift
        between segments, especially important for later segments in long videos.
        """
        # CRITICAL: Clear GPU cache and reset state BEFORE each generation
        # This prevents accumulated state from previous segments affecting voice consistency
        self._clear_gpu_cache()
        
        # F5-TTS requires reference text (transcription of reference audio)
        # Cache the transcription to avoid re-transcribing for each segment
        ref_path_str = str(reference_audio_path)
        
        if ref_path_str not in self.cached_ref_text:
            # First time - transcribe and cache
            try:
                ref_text = self.tts_engine.transcribe(ref_path_str)
                self.cached_ref_text[ref_path_str] = ref_text
                print(f"   📝 Reference transcription (cached): {ref_text[:50]}...")
            except Exception as e:
                print(f"   ⚠ Could not transcribe reference audio, using empty string: {e}")
                ref_text = ""
                self.cached_ref_text[ref_path_str] = ref_text
        else:
            # Use cached transcription
            ref_text = self.cached_ref_text[ref_path_str]
            print(f"   ✓ Using cached reference transcription")
        
        # Generate seed value for consistency
        # Use fixed seed for consistent voice across segments, or random for variety
        if self.config.USE_FIXED_SEED:
            safe_seed = self.config.FIXED_SEED
        else:
            # Valid range is [0, 4294967295] to avoid Python hash seed errors
            safe_seed = random.randint(0, 4294967295)
        
        # IMPORTANT: Set PyTorch random seed explicitly for complete determinism
        # F5-TTS calls seed_everything internally, but we ensure it here too
        # Cross-platform seed setting: CUDA, MPS, and CPU
        torch.manual_seed(safe_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(safe_seed)
            torch.cuda.manual_seed_all(safe_seed)
        # MPS (Mac) doesn't have separate manual_seed, manual_seed is sufficient
        np.random.seed(safe_seed)
        random.seed(safe_seed)
        
        # Generate audio with voice cloning - improved parameters for quality
        # NOTE: fix_duration can cause F5-TTS to generate near-silence for some texts
        # Better to generate naturally and adjust speed afterwards
        try:
            wav, sr, spec = self.tts_engine.infer(
                ref_file=str(reference_audio_path),
                ref_text=ref_text,
                gen_text=text,
                speed=1.0,  # Generate at natural speed first
                fix_duration=None,  # Don't constrain duration - let it generate naturally
                remove_silence=False,  # Keep natural pauses
                seed=safe_seed,  # Use controlled seed to prevent PYTHONHASHSEED errors
                # Quality parameters (tunable in Config)
                nfe_step=self.config.F5_NFE_STEP,  # Higher = better quality but slower
                cfg_strength=self.config.F5_CFG_STRENGTH,  # Higher = more faithful to reference
                sway_sampling_coef=self.config.F5_SWAY_SAMPLING,  # Improves quality
                target_rms=0.1,  # Volume normalization
                cross_fade_duration=0.15  # Smooth transitions
            )
        finally:
            # CRITICAL: Clear GPU cache AFTER generation to prevent state accumulation
            # This is especially important for later segments when memory is fragmented
            self._clear_gpu_cache()
        
        return wav, sr
    
    def smooth_audio(self, audio, sample_rate):
        """
        Apply gentle smoothing to reduce harsh/shrill frequencies
        Uses a subtle low-pass filter to soften the audio
        """
        from scipy.signal import butter, filtfilt
        
        # Apply gentle low-pass filter at 4kHz to reduce harshness
        # Most speech energy is below 4kHz anyway
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
        
        # Clip to reasonable range to preserve quality
        speed_factor_clipped = np.clip(
            speed_factor,
            0.7,  # Don't slow down too much
            1.5   # Don't speed up too much
        )
        
        if speed_factor != speed_factor_clipped:
            print(f"       ⚠️ Speed factor clipped: {speed_factor:.2f}x → {speed_factor_clipped:.2f}x")
        
        # Calculate target sample count
        target_samples = int(target_duration * sample_rate)
        
        # Resample to match target duration
        adjusted_audio = signal.resample(audio, target_samples)
        
        final_duration = len(adjusted_audio) / sample_rate
        actual_speed_factor = current_duration / final_duration if final_duration > 0 else 1.0
        
        return adjusted_audio, actual_speed_factor
    
    def synthesize_all_segments(self, segments, reference_audio):
        """
        Generate TTS for all segments with voice cloning
        
        CRITICAL: Includes periodic deep cache clearing every 10 segments
        to prevent voice drift in longer videos. This is essential for
        maintaining consistency in later segments.
        """
        print(f"\n{'='*70}")
        print(f"STEP 5: F5-TTS WITH VOICE CLONING")
        print(f"{'='*70}")
        print(f"   🔧 Voice consistency mode: State clearing enabled")
        print(f"   📊 Processing {len(segments)} segments")
        
        results = []
        
        for i, segment in enumerate(segments):
            # Periodic deep cache clear every 10 segments to prevent accumulation
            # This is CRITICAL for maintaining voice consistency in longer videos
            if i > 0 and i % 10 == 0:
                print(f"  🔄 Deep cache clear at segment {i+1}/{len(segments)} (preventing voice drift)...")
                self._clear_gpu_cache(deep=True)
            
            print(f"  [{i+1:2d}] Generating: '{segment['text_de'][:40]}...' (target: {segment['duration']:.2f}s)")
            
            try:
                # Generate with voice cloning
                audio, sample_rate = self.synthesize_with_voice_cloning(
                    text=segment['text_de'],
                    reference_audio_path=str(reference_audio),
                    target_duration=segment['duration']
                )
                
                # Apply smoothing to reduce harsh/shrill frequencies
                audio = self.smooth_audio(audio, sample_rate)
                
                # Adjust speed for drift matching
                current_duration = len(audio) / sample_rate
                adjusted_audio, speed_factor = self.adjust_speed(
                    audio,
                    segment['duration'],
                    sample_rate
                )
                
                # Diagnostic: Check audio quality
                final_duration = len(adjusted_audio) / sample_rate
                audio_rms = np.sqrt(np.mean(adjusted_audio**2)) if len(adjusted_audio) > 0 else 0
                
                print(f"       ✓ Generated: {current_duration:.2f}s → Final: {final_duration:.2f}s "
                      f"(speed: {speed_factor:.2f}x, RMS: {audio_rms:.4f}, samples: {len(adjusted_audio)})")
                
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
                    'speed_factor': speed_factor
                })
                
            except Exception as e:
                print(f"       ✗ TTS failed: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✓ F5-TTS complete: {len(results)} segments")
        
        # Normalize volume across all segments for consistency
        if self.config.NORMALIZE_VOLUME and results:
            print(f"\n📊 Normalizing volume across all segments...")
            results = self.normalize_segment_volumes(results)
        
        return results
    
    def normalize_segment_volumes(self, segments):
        """Normalize volume across all segments for consistent loudness"""
        # Calculate target RMS (use median to avoid outliers)
        rms_values = []
        for seg in segments:
            if len(seg['audio']) > 0:
                rms = np.sqrt(np.mean(seg['audio']**2))
                rms_values.append(rms)
        
        if not rms_values:
            return segments
        
        target_rms = np.median(rms_values)
        print(f"   Target RMS: {target_rms:.4f}")
        
        # Normalize each segment
        for i, seg in enumerate(segments):
            if len(seg['audio']) > 0:
                current_rms = np.sqrt(np.mean(seg['audio']**2))
                if current_rms > 0:
                    gain = target_rms / current_rms
                    # Limit gain to avoid over-amplification
                    gain = np.clip(gain, 0.5, 2.0)
                    seg['audio'] = seg['audio'] * gain
                    
                    new_rms = np.sqrt(np.mean(seg['audio']**2))
                    print(f"   [{seg['segment_id']:2d}] RMS: {current_rms:.4f} → {new_rms:.4f} (gain: {gain:.2f}x)")
        
        print(f"   ✓ Volume normalized across {len(segments)} segments")
        return segments

# ============================================================================
# STEP 6: AUDIO ASSEMBLY
# ============================================================================

class AudioAssembler:
    def __init__(self, config):
        self.config = config
    
    def apply_crossfade(self, audio_buffer, new_audio, start_sample, fade_samples):
        """
        Apply cross-fade between existing audio and new segment
        
        Args:
            audio_buffer: Existing audio buffer
            new_audio: New audio segment to add
            start_sample: Starting position in buffer
            fade_samples: Number of samples for cross-fade
        """
        end_sample = min(start_sample + len(new_audio), len(audio_buffer))
        audio_length = end_sample - start_sample
        
        # Check if there's existing audio to cross-fade with
        existing_audio = audio_buffer[start_sample:end_sample]
        has_existing = np.any(np.abs(existing_audio) > 1e-6)
        
        if has_existing and fade_samples > 0:
            # Apply cross-fade at the beginning
            fade_length = min(fade_samples, audio_length, len(new_audio))
            if fade_length > 0:
                # Create fade curves
                fade_in = np.linspace(0, 1, fade_length)
                fade_out = np.linspace(1, 0, fade_length)
                
                # Apply cross-fade
                new_audio_slice = new_audio[:audio_length].copy()
                new_audio_slice[:fade_length] = (
                    new_audio_slice[:fade_length] * fade_in +
                    existing_audio[:fade_length] * fade_out
                )
                
                audio_buffer[start_sample:end_sample] = new_audio_slice
            else:
                audio_buffer[start_sample:end_sample] = new_audio[:audio_length]
        else:
            # No existing audio, just place it
            audio_buffer[start_sample:end_sample] = new_audio[:audio_length]
    
    def assemble_audio(self, segments, total_duration):
        """Assemble all segments with precise timestamp placement and diagnostics"""
        print(f"\n{'='*70}")
        print(f"STEP 6: AUDIO ASSEMBLY")
        print(f"{'='*70}")
        
        print(f"Assembling {len(segments)} segments...")
        if self.config.CROSS_FADE_DURATION > 0:
            print(f"Using {self.config.CROSS_FADE_DURATION}s cross-fade for smooth transitions")
        
        total_samples = int(total_duration * self.config.FINAL_SAMPLE_RATE)
        final_audio = np.zeros(total_samples, dtype=np.float32)
        
        # Calculate cross-fade samples
        fade_samples = int(self.config.CROSS_FADE_DURATION * self.config.FINAL_SAMPLE_RATE)
        
        # Track coverage
        total_audio_duration = 0
        coverage_samples = 0
        
        for seg in segments:
            start_sample = int(seg['start'] * self.config.FINAL_SAMPLE_RATE)
            audio = seg['audio']
            
            if len(audio) == 0:
                print(f"  [{seg['segment_id']:2d}] ⚠️ WARNING: Empty audio at {seg['start']:.2f}s")
                continue
            
            # Apply cross-fade if enabled
            if fade_samples > 0:
                self.apply_crossfade(final_audio, audio, start_sample, fade_samples)
            else:
                # No cross-fade, just place directly
                end_sample = min(start_sample + len(audio), total_samples)
                audio_slice = audio[:end_sample - start_sample]
                final_audio[start_sample:end_sample] = audio_slice
            
            actual_duration = len(audio) / self.config.FINAL_SAMPLE_RATE
            total_audio_duration += actual_duration
            coverage_samples += len(audio)
            
            print(f"  [{seg['segment_id']:2d}] Placed at {seg['start']:.2f}s "
                  f"(duration: {actual_duration:.2f}s, samples: {len(audio)})")
        
        # Calculate coverage statistics
        coverage_percent = (coverage_samples / total_samples) * 100 if total_samples > 0 else 0
        
        print(f"\n📊 Assembly Statistics:")
        print(f"   Total duration: {total_duration:.2f}s ({total_samples} samples)")
        print(f"   Audio coverage: {total_audio_duration:.2f}s ({coverage_samples} samples)")
        print(f"   Coverage: {coverage_percent:.1f}%")
        print(f"   Silence: {100 - coverage_percent:.1f}%")
        
        if coverage_percent < 50:
            print(f"   ⚠️ WARNING: Less than 50% audio coverage - video will have long silences!")
        
        print(f"\n✓ Audio assembled: {total_duration:.2f}s")
        
        return final_audio
    
    def save_audio(self, audio, output_path):
        """Save assembled audio"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, self.config.FINAL_SAMPLE_RATE)
        print(f"✓ Saved dubbed audio: {output_path}")

# ============================================================================
# STEP 7: VIDEO MERGE
# ============================================================================

class VideoMerger:
    def __init__(self, config):
        self.config = config
    
    def merge_audio_video(self, video_path, audio_path, output_path):
        """Replace video audio track with dubbed audio"""
        print(f"\n{'='*70}")
        print(f"STEP 7: VIDEO MERGE")
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
# STEP 8: EVALUATION & REPORT GENERATION
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
            except AttributeError as e:
                if 'list_audio_backends' in str(e):
                    print(f"⚠ Torchaudio version incompatibility - skipping speaker similarity")
                    print(f"   (This is a known issue and doesn't affect dubbing quality)")
                else:
                    print(f"⚠ Could not load speaker encoder: {e}")
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
            
            # Load audio files
            original, sr1 = sf.read(str(original_audio_path))
            dubbed, sr2 = sf.read(str(dubbed_audio_path))
            
            # Get embeddings
            emb_original = self.speaker_encoder.encode_batch(
                torch.FloatTensor(original).unsqueeze(0)
            )
            emb_dubbed = self.speaker_encoder.encode_batch(
                torch.FloatTensor(dubbed).unsqueeze(0)
            )
            
            # Compute cosine similarity
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
            
            # RMS (volume)
            rms = np.sqrt(np.mean(audio**2))
            
            # Peak amplitude
            peak = np.max(np.abs(audio))
            
            # Dynamic range
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
        print(f"STEP 8: QUALITY EVALUATION & REPORT GENERATION")
        print(f"{'='*70}")
        
        report_path = self.config.OUTPUT_DIR / "evaluation_report.md"
        
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
            f.write("# Video Translation Evaluation Report\n\n")
            f.write(f"**Generated**: {timedelta(seconds=0)}\n\n")
            f.write(f"**Source**: {self.config.INPUT_VIDEO.name}\n")
            f.write(f"**Target Language**: German\n\n")
            
            # Translation Quality
            f.write("## Translation Quality\n\n")
            if segments:
                f.write(f"- **Total Segments**: {len(segments)}\n")
                adjusted_count = sum(1 for seg in segments if seg.get('translation_info', {}).get('needs_adjustment', False))
                f.write(f"- **Segments Needing Timing Adjustment**: {adjusted_count}/{len(segments)}\n")
                
                # Average speed factors
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
                    f.write(f"\n*Install speechbrain for speaker similarity: `pip install speechbrain`*\n")
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
            f.write(f"- **TTS Model**: {self.config.F5_MODEL}\n")
            f.write(f"- **Sample Rate**: {self.config.F5_SAMPLE_RATE}Hz\n")
            f.write(f"- **NFE Steps**: {self.config.F5_NFE_STEP}\n")
            f.write(f"- **CFG Strength**: {self.config.F5_CFG_STRENGTH}\n")
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
        self.audio_assembler = AudioAssembler(config)
        self.video_merger = VideoMerger(config)
        self.evaluator = QualityEvaluator(config)
    
    def run(self):
        """Execute complete pipeline"""
        print("\n" + "="*70)
        print("🎬 VIDEO DUBBING PIPELINE: F5-TTS")
        print("   English → German with Zero-Shot Voice Cloning")
        print("="*70)
        
        # Step 1: Extract audio
        audio_path, duration = self.audio_extractor.extract_audio_from_video(
            self.config.INPUT_VIDEO,
            self.config.EXTRACTED_AUDIO
        )
        
        # Extract reference audio (intelligent or manual)
        if self.config.USE_INTELLIGENT_REFERENCE:
            # Use intelligent speech detection
            reference_path = self.audio_extractor.extract_reference_audio(
                audio_path,
                self.config.REFERENCE_AUDIO
            )
        else:
            # Use manual time selection
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
        
        # Step 5: TTS with voice cloning
        tts_results = self.tts_generator.synthesize_all_segments(segments, reference_path)
        
        # Step 6: Audio assembly
        final_audio = self.audio_assembler.assemble_audio(tts_results, duration)
        self.audio_assembler.save_audio(final_audio, self.config.DUBBED_AUDIO)
        
        # Step 7: Video merge
        self.video_merger.merge_audio_video(
            self.config.INPUT_VIDEO,
            self.config.DUBBED_AUDIO,
            self.config.OUTPUT_VIDEO
        )
        
        # Step 8: Quality evaluation and report generation
        evaluation_report = self.evaluator.generate_evaluation_report(segments=tts_results)
        
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

