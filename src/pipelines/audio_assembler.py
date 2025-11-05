"""
Audio assembly module
"""
from pathlib import Path

import numpy as np
import soundfile as sf
import torchaudio
from scipy import signal

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
        print(f"STEP 6: AUDIO ASSEMBLY")
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
                print(f"  [{seg['segment_id']:2d}] WARNING: Empty audio at {seg['start']:.2f}s")
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
            print(f"\nUpsampling from {sample_rate}Hz to {self.config.FINAL_SAMPLE_RATE}Hz...")
            final_audio = self._resample(final_audio, sample_rate, self.config.FINAL_SAMPLE_RATE)
            sample_rate = self.config.FINAL_SAMPLE_RATE
        
        # Calculate coverage statistics
        coverage_percent = (coverage_samples / total_samples) * 100 if total_samples > 0 else 0
        silence_percent = 100 - coverage_percent
        
        # Compile assembly statistics
        assembly_stats = {
            'total_segments': len(segments),
            'total_duration': total_duration,
            'audio_coverage_duration': total_audio_duration,
            'coverage_percent': coverage_percent,
            'silence_percent': silence_percent,
            'total_samples': total_samples,
            'coverage_samples': coverage_samples,
            'sample_rate': sample_rate,
            'cross_fade_duration': self.config.CROSS_FADE_DURATION,
            'has_low_coverage': coverage_percent < 50
        }
        
        print(f"\nAssembly Statistics:")
        print(f"Total duration: {total_duration:.2f}s")
        print(f"Audio coverage: {total_audio_duration:.2f}s")
        print(f"Coverage: {coverage_percent:.1f}%")
        print(f"Silence: {silence_percent:.1f}%")
        
        if coverage_percent < 50:
            print(f"WARNING: Less than 50% audio coverage - video will have long silences")
        
        print(f"Audio assembled: {total_duration:.2f}s")
        
        return final_audio, sample_rate, assembly_stats
    
    def save_audio(self, audio, output_path, sample_rate):
        """Save assembled audio"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, sample_rate)
        print(f"Saved dubbed audio: {output_path}")

# ============================================================================
# STEP 8: VIDEO MERGE
# ============================================================================

