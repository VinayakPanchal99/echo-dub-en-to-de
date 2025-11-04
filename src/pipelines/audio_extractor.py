"""
Audio extraction module
"""
import subprocess
import soundfile as sf
import numpy as np
from pathlib import Path


class AudioExtractor:
    def __init__(self, config):
        self.config = config
    
    def extract_audio_from_video(self, video_path, output_path, sample_rate=24000):
        """Extract full audio track from video"""
        print(f"\n{'='*70}")
        print(f"STEP 1: AUDIO EXTRACTION")
        print(f"{'='*70}")
        
        output_path = Path(output_path)
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
        print(f"Extracted audio: {output_path}")
        
        audio, sr = sf.read(str(output_path))
        duration = len(audio) / sr
        print(f"Duration: {duration:.2f}s, Sample Rate: {sr}Hz")
        
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
        print(f"\nSelecting reference audio...")
        print(f"Analyzing audio for continuous speech segments...")
        
        audio, sr = sf.read(str(audio_path))
        
        # Detect speech segments
        segments = self.detect_speech_segments(audio, sr)
        
        if not segments:
            print(f"WARNING: No speech segments detected, using default position")
            return self.config.REF_START_TIME, self.config.REF_DURATION, 0.0
        
        print(f"Found {len(segments)} speech segments")
        
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
            print(f"No single segment found, trying to combine segments...")
            
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
            print(f"Best segment: {start:.2f}s - {start+duration:.2f}s")
            print(f"Duration: {duration:.2f}s, Energy: {energy:.4f}, Score: {score:.2f}")
            return start, duration, score
        else:
            print(f"WARNING: No suitable segment found, using default")
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
            print(f"\nExtracting reference audio...")
            print(f"Using manual selection: Start {start_time}s, Duration {duration}s")
        
        audio, sr = sf.read(str(audio_path))
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)
        reference = audio[start_sample:end_sample]
        
        # Verify quality
        rms = np.sqrt(np.mean(reference**2))
        print(f"Audio quality - RMS: {rms:.4f}, Samples: {len(reference)}")
        
        if rms < 0.01:
            print(f"WARNING: Low audio energy, may affect voice cloning quality")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), reference, sr)
        print(f"Reference audio saved: {output_path}")
        
        return output_path

