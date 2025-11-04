"""
Prosody analysis module
"""
import json
from pathlib import Path

import numpy as np
import parselmouth

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
            print("Prosody analysis disabled (skipping)")
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
                print(f"  [{i+1:2d}] ERROR: Prosody analysis failed: {e}")
                segment['prosody'] = None
        
        self.config.PROSODY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.config.PROSODY_OUTPUT), 'w') as f:
            json.dump({'segments': [{'index': seg['index'], 'prosody': seg.get('prosody')} for seg in segments]}, f, indent=2)
        
        print(f"Prosody analysis complete")
        print(f"Data saved to: {self.config.PROSODY_OUTPUT}")
        
        return segments

# ============================================================================
# STEP 4: TRANSLATION (NLLB-200)
# ============================================================================

