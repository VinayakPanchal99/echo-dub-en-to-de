"""
Video merging module
"""
import subprocess
from pathlib import Path

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
        
        print(f"Final video saved: {output_path}")

# ============================================================================
# STEP 9: EVALUATION & REPORT GENERATION
# ============================================================================

