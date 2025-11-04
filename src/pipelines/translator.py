"""
Translation module using NLLB-200
"""
from datetime import timedelta
from pathlib import Path
import time

import srt
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translator:
    def __init__(self, config):
        self.config = config
        print(f"\nInitializing NLLB-200 translator...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.NLLB_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.NLLB_MODEL)
        self.model.to(config.DEVICE)
        self.model.eval()
        
        print(f"NLLB-200 model loaded on {config.DEVICE}")
    
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
        
        # Model inference
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
        print(f"STEP 3: TRANSLATION (NLLB-200)")
        print(f"{'='*70}")
        
        print(f"Translating {len(segments)} segments from English to German...")
        
        total_start = time.time()
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
            except Exception as e:
                print(f"  ERROR: Translation failed for segment {i+1}: {e}")
                segment['text_de'] = segment['text_en']
                segment['translation_info'] = None
        
        total_elapsed = time.time() - total_start
        print(f"Total time: {total_elapsed:.2f}s ({total_elapsed/len(segments):.2f}s per segment avg)")
        
        self._save_translated_srt(segments)
        
        print(f"Translation complete")
        print(f"Translated SRT saved: {self.config.TRANSLATED_SRT}")
        
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

