"""
Quality evaluation module
"""
import numpy as np
import soundfile as sf
from typing import List, Dict, Optional, Tuple

class QualityEvaluator:
    def __init__(self, config):
        self.config = config
        self.speaker_encoder = None
        
    def _load_speaker_encoder(self):
        """Load speaker verification model for similarity computation using Resemblyzer"""
        if self.speaker_encoder is None:
            try:
                from resemblyzer import VoiceEncoder
                print(f"\nLoading speaker verification model (Resemblyzer)...")
                self.speaker_encoder = VoiceEncoder()
                print(f"Speaker encoder loaded")
            except ImportError:
                print(f"WARNING: Resemblyzer not installed")
                print(f"Install with: pip install resemblyzer")
                self.speaker_encoder = None
            except Exception as e:
                print(f"WARNING: Could not load speaker encoder: {e}")
                self.speaker_encoder = None
    
    def compute_speaker_similarity(self, original_audio_path, dubbed_audio_path):
        """Compute cosine similarity between original and dubbed audio using Resemblyzer"""
        try:
            self._load_speaker_encoder()
            
            if self.speaker_encoder is None:
                return None, None, "Speaker encoder not available"
            
            from resemblyzer import preprocess_wav
            
            # Load and preprocess audio files
            original_wav = preprocess_wav(str(original_audio_path))
            dubbed_wav = preprocess_wav(str(dubbed_audio_path))
            
            # Get embeddings
            emb_original = self.speaker_encoder.embed_utterance(original_wav)
            emb_dubbed = self.speaker_encoder.embed_utterance(dubbed_wav)
            
            # Compute cosine similarity
            similarity = np.dot(emb_original, emb_dubbed) / (
                np.linalg.norm(emb_original) * np.linalg.norm(emb_dubbed)
            )
            
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
    
    def compute_translation_metrics(self, segments: List[Dict]) -> Dict:
        """Compute translation quality metrics"""
        if not segments:
            return {}
        
        metrics = {
            'total_segments': len(segments),
            'avg_en_length': 0,
            'avg_de_length': 0,
            'avg_length_ratio': 0,
            'avg_en_words': 0,
            'avg_de_words': 0,
            'avg_word_ratio': 0
        }
        
        en_lengths = []
        de_lengths = []
        en_word_counts = []
        de_word_counts = []
        
        for seg in segments:
            en_text = seg.get('text_en', '')
            de_text = seg.get('text_de', '')
            
            if en_text and de_text:
                en_lengths.append(len(en_text))
                de_lengths.append(len(de_text))
                en_word_counts.append(len(en_text.split()))
                de_word_counts.append(len(de_text.split()))
        
        if en_lengths:
            metrics['avg_en_length'] = np.mean(en_lengths)
            metrics['avg_de_length'] = np.mean(de_lengths)
            metrics['avg_length_ratio'] = np.mean([de/en if en > 0 else 0 for de, en in zip(de_lengths, en_lengths)])
            metrics['avg_en_words'] = np.mean(en_word_counts)
            metrics['avg_de_words'] = np.mean(de_word_counts)
            metrics['avg_word_ratio'] = np.mean([de/en if en > 0 else 0 for de, en in zip(de_word_counts, en_word_counts)])
        
        return metrics
    
    def generate_evaluation_report(self, segments=None, assembly_stats=None):
        """Generate comprehensive evaluation report"""
        print(f"\n{'='*70}")
        print(f"STEP 8: QUALITY EVALUATION & REPORT GENERATION")
        print(f"{'='*70}")
        
        report_path = self.config.OUTPUT_DIR / "evaluation_report_seedvc.md"
        
        # Compute metrics
        print("Computing timing alignment...")
        orig_dur, dubbed_dur, drift, drift_pct = self.compute_timing_alignment(
            self.config.EXTRACTED_AUDIO,
            self.config.DUBBED_AUDIO
        )
        
        # Compute speaker similarity
        print("Computing speaker similarity...")
        similarity, segment_mean, similarity_error = self.compute_speaker_similarity(
            self.config.EXTRACTED_AUDIO,
            self.config.DUBBED_AUDIO
        )
        
        # Compute translation metrics
        translation_metrics = {}
        if segments:
            print("Computing translation metrics...")
            translation_metrics = self.compute_translation_metrics(segments)
        
        # Generate markdown report
        with open(str(report_path), 'w', encoding='utf-8') as f:
            f.write("# Video Translation Evaluation Report\n\n")
            f.write(f"**Source**: {self.config.INPUT_VIDEO.name}\n")
            f.write(f"**Target Language**: German\n\n")
            
            # Translation Quality Metrics
            if translation_metrics:
                f.write("## Translation Quality (EN → DE)\n\n")
                f.write(f"- **Total Segments**: {translation_metrics.get('total_segments', 0)}\n")
                f.write(f"- **Avg English Words**: {translation_metrics.get('avg_en_words', 0):.1f}\n")
                f.write(f"- **Avg German Words**: {translation_metrics.get('avg_de_words', 0):.1f}\n")
                f.write(f"- **Word Ratio (DE/EN)**: {translation_metrics.get('avg_word_ratio', 0):.2f}x\n")
                f.write("\n")
            
            # Assembly Statistics
            if assembly_stats:
                f.write("## Audio Assembly Statistics\n\n")
                f.write(f"- **Total Segments Processed**: {assembly_stats.get('total_segments', 0)}\n")
                f.write(f"- **Total Duration**: {assembly_stats.get('total_duration', 0):.2f}s\n")
                f.write(f"- **Audio Coverage Duration**: {assembly_stats.get('audio_coverage_duration', 0):.2f}s\n")
                f.write(f"- **Audio Coverage**: {assembly_stats.get('coverage_percent', 0):.1f}%\n")
                f.write(f"- **Silence Percentage**: {assembly_stats.get('silence_percent', 0):.1f}%\n")
                f.write(f"- **Sample Rate**: {assembly_stats.get('sample_rate', 0)} Hz\n")
                f.write(f"- **Cross-Fade Duration**: {assembly_stats.get('cross_fade_duration', 0):.3f}s\n")
                coverage_status = '✅ PASS' if assembly_stats.get('coverage_percent', 0) >= 50 else '⚠️ WARNING'
                f.write(f"- **Coverage Status**: {coverage_status}\n")
                if assembly_stats.get('has_low_coverage', False):
                    f.write(f"  - ⚠️ Low coverage detected - video may have long silences\n")
                f.write("\n")
            
            # Speaker Similarity
            f.write("## Speaker Similarity\n\n")
            if similarity is not None:
                f.write(f"- **Cosine Similarity**: {similarity:.4f}\n")
                f.write(f"- **Threshold (≥0.65)**: {'✅ PASS' if similarity >= 0.65 else '❌ FAIL'}\n")
            else:
                f.write(f"- **Status**: ⚠️ Not computed\n")
                if similarity_error:
                    f.write(f"- **Reason**: {similarity_error}\n")
            f.write("\n")
            
            # Timing Alignment
            if orig_dur is not None:
                f.write("## Timing Alignment\n\n")
                f.write(f"- **Duration**: {orig_dur:.2f}s → {dubbed_dur:.2f}s\n")
                f.write(f"- **Drift**: {drift:+.2f}s ({drift_pct:+.1f}%)\n")
                f.write(f"- **Status**: {'✅ PASS' if abs(drift_pct) <= 5 else '❌ FAIL'}\n")
            f.write("\n")
        
        print(f"Evaluation report saved: {report_path}")
        return report_path

# ============================================================================
# MAIN PIPELINE
# ============================================================================

