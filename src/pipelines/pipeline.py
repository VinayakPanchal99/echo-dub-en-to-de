"""
Main video dubbing pipeline
"""
from pathlib import Path

from src.pipelines.audio_extractor import AudioExtractor
from src.pipelines.srt_parser import SRTParser
from src.pipelines.translator import Translator
from src.pipelines.tts_synthesizer import F5TTSSynthesizer
from src.pipelines.audio_assembler import AudioAssembler
from src.pipelines.video_merger import VideoMerger
from src.pipelines.evaluator import QualityEvaluator

from src.utils.paths import setup_paths

# Setup paths
setup_paths()

# Try to import SeedVCVoiceConverter to check availability
try:
    from src.pipelines.voice_converter import SeedVCVoiceConverter, SEED_VC_MODULES_AVAILABLE
    SEED_VC_AVAILABLE = SEED_VC_MODULES_AVAILABLE
except (ImportError, AttributeError):
    SeedVCVoiceConverter = None
    SEED_VC_AVAILABLE = False

class VideoDubbingPipeline:
    def __init__(self, config):
        self.config = config
        
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        self.audio_extractor = AudioExtractor(config)
        self.srt_parser = SRTParser(config)
        self.translator = Translator(config)
        self.tts_generator = F5TTSSynthesizer(config)
        
        # Initialize voice converter if available
        if SEED_VC_AVAILABLE and SeedVCVoiceConverter is not None:
            self.voice_converter = SeedVCVoiceConverter(config)
        else:
            self.voice_converter = None
            
        self.audio_assembler = AudioAssembler(config)
        self.video_merger = VideoMerger(config)
        self.evaluator = QualityEvaluator(config)
    
    def run(self):
        """Execute complete pipeline"""
        print("\n" + "="*70)
        print("VIDEO DUBBING PIPELINE: F5-TTS + Seed-VC")
        print("English -> German")
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
        
        # Step 3: Translation
        segments = self.translator.translate_all_segments(segments)
        
        # Step 4: TTS without voice cloning (basic generation)
        tts_results = self.tts_generator.synthesize_all_segments(segments, reference_path)
        
        # Step 5: Voice conversion with Seed-VC
        if self.voice_converter is None:
            raise RuntimeError("Seed-VC voice converter is not available. Please check Seed-VC installation.")
        
        vc_results = self.voice_converter.convert_all_segments(tts_results, reference_path)
        
        # Step 6: Audio assembly
        final_audio, sample_rate, assembly_stats = self.audio_assembler.assemble_audio(vc_results, duration)
        self.audio_assembler.save_audio(final_audio, self.config.DUBBED_AUDIO, sample_rate)
        
        # Step 7: Video merge
        self.video_merger.merge_audio_video(
            self.config.INPUT_VIDEO,
            self.config.DUBBED_AUDIO,
            self.config.OUTPUT_VIDEO
        )
        
        # Step 8: Quality evaluation and report generation
        evaluation_report = self.evaluator.generate_evaluation_report(
            segments=vc_results,
            assembly_stats=assembly_stats
        )

# ============================================================================
# ENTRY POINT
# ============================================================================

