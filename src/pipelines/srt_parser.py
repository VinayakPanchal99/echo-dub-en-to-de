"""
SRT subtitle parsing module
"""
import srt


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
        
        print(f"Parsed {len(segments)} segments from SRT")
        print(f"Total duration: {segments[-1]['end']:.2f}s")
        
        return segments

