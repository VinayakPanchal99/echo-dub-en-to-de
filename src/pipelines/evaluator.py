"""
Quality evaluation module
"""
import json
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Suppress SpeechBrain warnings that are not critical
# These are deprecation warnings from SpeechBrain's internal code
warnings.filterwarnings('ignore', message='.*torch\.cuda\.amp\.custom_fwd.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*weights_only.*', category=FutureWarning)

class QualityEvaluator:
    def __init__(self, config):
        self.config = config
        self.speaker_encoder = None
        
    def _load_speaker_encoder(self):
        """Load speaker verification model for similarity computation"""
        if self.speaker_encoder is None:
            try:
                # Suppress SpeechBrain warnings during model loading
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    warnings.filterwarnings('ignore', message='.*torch\.cuda\.amp\.custom_fwd.*', category=FutureWarning)
                    warnings.filterwarnings('ignore', message='.*weights_only.*', category=FutureWarning)
                    
                    from speechbrain.pretrained import EncoderClassifier
                    print(f"\nLoading speaker verification model...")
                    self.speaker_encoder = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        savedir="temp/speaker_encoder"
                    )
                print(f"Speaker encoder loaded")
            except ImportError:
                print(f"WARNING: SpeechBrain not installed")
                print(f"Install with: pip install speechbrain")
                self.speaker_encoder = None
            except Exception as e:
                print(f"WARNING: Could not load speaker encoder: {e}")
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
            print(f"WARNING: Could not compute timing alignment: {e}")
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
            print(f"WARNING: Could not compute audio quality: {e}")
            return None
    
    def generate_evaluation_report(self, segments=None):
        """Generate comprehensive evaluation report"""
        print(f"\n{'='*70}")
        print(f"STEP 8: QUALITY EVALUATION & REPORT GENERATION")
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
        
        print(f"Evaluation report saved: {report_path}")
        return report_path

# ============================================================================
# MAIN PIPELINE
# ============================================================================

