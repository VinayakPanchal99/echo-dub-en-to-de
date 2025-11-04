"""
Seed-VC voice conversion module
"""
import os
import warnings
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
import yaml

# Suppress Seed-VC warnings that are not critical
# These warnings come from Seed-VC's internal code and PyTorch deprecations
# Suppress globally since they come from external Seed-VC modules
warnings.filterwarnings('ignore', message='.*weight_norm.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*weights_only.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*torch\.load.*weights_only.*', category=FutureWarning)
# Note: Shape mismatch warnings are kept visible as they're informative about model loading

# Try to import Seed-VC modules
try:
    from hf_utils import load_custom_model_from_hf
    from modules.audio import mel_spectrogram
    from modules.bigvgan import bigvgan
    from modules.campplus.DTDNN import CAMPPlus
    from modules.commons import build_model, load_checkpoint, recursive_munch
    SEED_VC_MODULES_AVAILABLE = True
except ImportError as e:
    SEED_VC_MODULES_AVAILABLE = False
    _seed_vc_import_error = e


class SeedVCVoiceConverter:
    def __init__(self, config):
        self.config = config
        
        if not SEED_VC_MODULES_AVAILABLE:
            error_msg = "Seed-VC is not available. Please:\n"
            error_msg += "1. Clone Seed-VC: git clone https://github.com/Plachtaa/seed-vc.git external/seed-vc\n"
            error_msg += "2. Install dependencies as per Seed-VC README"
            if '_seed_vc_import_error' in globals():
                error_msg += f"\n\nImport error: {_seed_vc_import_error}"
            raise ImportError(error_msg)
        
        if not config.SEED_VC_CONFIG or not config.SEED_VC_CONFIG.exists():
            raise FileNotFoundError(
                f"Seed-VC config not found at {config.SEED_VC_CONFIG}\n"
                "Please ensure Seed-VC is properly cloned and configured."
            )
        
        print(f"\nInitializing Seed-VC...")
        
        # Suppress warnings during Seed-VC initialization (weight_norm, torch.load deprecations)
        # These warnings come from Seed-VC's internal code and are not critical
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*weight_norm.*', category=FutureWarning)
            warnings.filterwarnings('ignore', message='.*weights_only.*', category=FutureWarning)
            
            # Load config
            with open(config.SEED_VC_CONFIG, 'r') as f:
                config_dict = yaml.safe_load(f)
            model_params = recursive_munch(config_dict['model_params'])
            model_params.dit_type = 'DiT'
            
            # Build model
            self.model = build_model(model_params, stage='DiT')
            
            # Load checkpoint
            checkpoint_path = self._get_checkpoint(config.SEED_VC_CHECKPOINT)
            self.model, _, _, _ = load_checkpoint(
                self.model,
                None,
                checkpoint_path,
                load_only_params=True,
                ignore_modules=[],
                is_distributed=False,
            )
            for key in self.model:
                self.model[key].eval()
                self.model[key].to(config.DEVICE)
            self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        
        # Store model params and config
        self.model_params = model_params
        self.config_dict = config_dict
        self.sr = config_dict['preprocess_params']['sr']
        self.hop_length = config_dict['preprocess_params']['spect_params']['hop_length']
        
        # Load additional modules (suppress warnings here too)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*weights_only.*', category=FutureWarning)
            self._load_whisper_model()
            self._load_campplus_model()
            self._load_vocoder()
        self._setup_mel_fn()
        
        print(f"Seed-VC loaded on {config.DEVICE}")
        print(f"Sample rate: {self.sr}Hz")
    
    def _load_whisper_model(self):
        """Load Whisper model for semantic feature extraction"""
        from transformers import AutoFeatureExtractor, WhisperModel
        
        whisper_name = self.model_params.speech_tokenizer.name if hasattr(self.model_params.speech_tokenizer, 'name') else "openai/whisper-small"
        print(f"  Loading Whisper model: {whisper_name}...")
        
        self.whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(self.config.DEVICE)
        del self.whisper_model.decoder  # Only need encoder
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)
        
        def semantic_fn(waves_16k):
            ori_inputs = self.whisper_feature_extractor(
                [waves_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000  # Explicitly specify sampling rate to avoid warnings
            )
            ori_input_features = self.whisper_model._mask_input_features(
                ori_inputs.input_features, 
                attention_mask=ori_inputs.attention_mask
            ).to(self.config.DEVICE)
            with torch.no_grad():
                ori_outputs = self.whisper_model.encoder(
                    ori_input_features.to(self.whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori
        
        self.semantic_fn = semantic_fn
        print("Whisper loaded")
    
    def _load_campplus_model(self):
        """Load CAMPPlus model for style feature extraction"""
        from modules.campplus.DTDNN import CAMPPlus
        from hf_utils import load_custom_model_from_hf
        
        print("  Loading CAMPPlus model...")
        campplus_ckpt_path = load_custom_model_from_hf(
            "funasr/campplus", "campplus_cn_common.bin", config_filename=None
        )
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        # Load checkpoint - using weights_only=False for model checkpoints (required for compatibility)
        # Suppress warning as this is intentional for loading model state dicts
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*weights_only.*', category=FutureWarning)
            checkpoint = torch.load(campplus_ckpt_path, map_location="cpu", weights_only=False)
        self.campplus_model.load_state_dict(checkpoint)
        self.campplus_model.eval()
        self.campplus_model.to(self.config.DEVICE)
        print("CAMPPlus loaded")
    
    def _load_vocoder(self):
        """Load BigVGAN vocoder"""
        from modules.bigvgan import bigvgan
        
        print("  Loading BigVGAN vocoder...")
        bigvgan_name = self.model_params.vocoder.name
        self.vocoder = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        self.vocoder.remove_weight_norm()
        self.vocoder = self.vocoder.eval().to(self.config.DEVICE)
        print("BigVGAN loaded")
    
    def _setup_mel_fn(self):
        """Setup mel spectrogram function"""
        from modules.audio import mel_spectrogram
        
        mel_fn_args = {
            "n_fft": self.config_dict['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.config_dict['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.config_dict['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.config_dict['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.sr,
            "fmin": 0,
            "fmax": None,
            "center": False
        }
        self.to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)
        
        # Setup overlap parameters
        self.overlap_frame_len = 16
        self.max_context_window = self.sr // self.hop_length * 30
        self.overlap_wave_len = self.overlap_frame_len * self.hop_length
    
    def _get_checkpoint(self, checkpoint_name):
        """Download or locate checkpoint"""
        from huggingface_hub import hf_hub_download
        
        checkpoint_path = os.path.join(
            os.path.expanduser("~/.cache/seed-vc"),
            checkpoint_name
        )
        
        if not os.path.exists(checkpoint_path):
            print(f"  Downloading {checkpoint_name} from HuggingFace...")
            checkpoint_path = hf_hub_download(
                repo_id="Plachta/Seed-VC",
                filename=checkpoint_name,
                cache_dir=os.path.expanduser("~/.cache/seed-vc")
            )
        
        return checkpoint_path
    
    def load_reference_audio(self, reference_path):
        """Load and prepare reference audio for voice conversion"""
        # Load audio - use default sample rate detection, but resample if needed
        audio, sr = torchaudio.load(str(reference_path))
        
        # Resample if needed
        if sr != self.sr:
            audio = torchaudio.functional.resample(
                audio, sr, self.sr
            )
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        self.reference_audio = audio
        print(f"Reference audio loaded: {reference_path}")
        print(f"Duration: {audio.shape[1] / self.sr:.2f}s")
    
    def _resample_audio(self, audio, orig_sr, target_sr):
        """Resample audio using torchaudio"""
        if orig_sr == target_sr:
            return audio
        
        # Convert numpy to torch tensor if needed
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
        else:
            audio_tensor = audio
        
        resampled = torchaudio.functional.resample(
            audio_tensor, orig_sr, target_sr
        )
        
        return resampled.squeeze().cpu().numpy() if isinstance(audio, np.ndarray) else resampled
    
    def convert_segment(self, audio, segment_id):
        """Apply voice conversion to one segment using Seed-VC"""
        try:
            # Convert audio to torch tensor if needed
            if isinstance(audio, np.ndarray):
                source_audio = torch.from_numpy(audio).float()
            else:
                source_audio = audio.float()
            
            if len(source_audio.shape) == 1:
                source_audio = source_audio.unsqueeze(0)
            
            # Resample to Seed-VC sample rate
            if source_audio.shape[-1] / self.config.F5_SAMPLE_RATE * self.sr != source_audio.shape[-1]:
                source_audio = torchaudio.functional.resample(
                    source_audio, self.config.F5_SAMPLE_RATE, self.sr
                )
            
            source_audio = source_audio.to(self.config.DEVICE)
            ref_audio = self.reference_audio.to(self.config.DEVICE)
            
            # Limit reference audio to 25 seconds
            ref_audio_limited = ref_audio[:, :self.sr * 25]
            
            # Resample to 16kHz for Whisper
            ref_waves_16k = torchaudio.functional.resample(ref_audio_limited, self.sr, 16000)
            converted_waves_16k = torchaudio.functional.resample(source_audio, self.sr, 16000)
            
            # Extract semantic features with Whisper
            if converted_waves_16k.size(-1) <= 16000 * 30:
                S_alt = self.semantic_fn(converted_waves_16k)
            else:
                # Handle long audio with overlapping chunks
                overlapping_time = 5
                S_alt_list = []
                buffer = None
                traversed_time = 0
                while traversed_time < converted_waves_16k.size(-1):
                    if buffer is None:
                        chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
                    else:
                        chunk = torch.cat([
                            buffer, 
                            converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]
                        ], dim=-1)
                    S_alt = self.semantic_fn(chunk)
                    if traversed_time == 0:
                        S_alt_list.append(S_alt)
                    else:
                        S_alt_list.append(S_alt[:, 50 * overlapping_time:])
                    buffer = chunk[:, -16000 * overlapping_time:]
                    traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
                S_alt = torch.cat(S_alt_list, dim=1)
            
            ori_waves_16k = torchaudio.functional.resample(ref_audio_limited, self.sr, 16000)
            S_ori = self.semantic_fn(ori_waves_16k)
            
            # Extract mel spectrograms
            mel = self.to_mel(source_audio.float())
            mel2 = self.to_mel(ref_audio_limited.float())
            
            target_lengths = torch.LongTensor([mel.size(2)]).to(mel.device)
            target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
            
            # Extract style features with CAMPPlus
            feat2 = torchaudio.compliance.kaldi.fbank(
                ref_waves_16k,
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000
            )
            feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
            style2 = self.campplus_model(feat2.unsqueeze(0))
            
            # Length regulation
            cond, _, _, _, _ = self.model.length_regulator(
                S_alt, ylens=target_lengths, n_quantizers=3, f0=None
            )
            prompt_condition, _, _, _, _ = self.model.length_regulator(
                S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
            )
            
            # Voice conversion with chunking
            max_source_window = self.max_context_window - mel2.size(2)
            processed_frames = 0
            generated_wave_chunks = []
            
            with torch.no_grad():
                while processed_frames < cond.size(1):
                    chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
                    is_last_chunk = processed_frames + max_source_window >= cond.size(1)
                    cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
                    
                    # MPS doesn't support autocast with float32, so disable it on MPS
                    device_type_str = self.config.DEVICE if isinstance(self.config.DEVICE, str) else str(self.config.DEVICE)
                    use_autocast = not device_type_str.startswith('mps')
                    
                    if use_autocast:
                        with torch.autocast(device_type=device_type_str, dtype=torch.float16):
                            # Voice conversion using CFM inference
                            vc_target = self.model.cfm.inference(
                                cat_condition,
                                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                                mel2, style2, None,
                                self.config.SEED_VC_DIFFUSION_STEPS,
                                inference_cfg_rate=self.config.SEED_VC_CFG_RATE
                            )
                            vc_target = vc_target[:, :, mel2.size(-1):]
                    else:
                        # No autocast on MPS - run in float32 directly
                        vc_target = self.model.cfm.inference(
                            cat_condition,
                            torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                            mel2, style2, None,
                            self.config.SEED_VC_DIFFUSION_STEPS,
                            inference_cfg_rate=self.config.SEED_VC_CFG_RATE
                        )
                        vc_target = vc_target[:, :, mel2.size(-1):]
                    
                    vc_wave = self.vocoder(vc_target.float())[0]
                    if vc_wave.ndim == 1:
                        vc_wave = vc_wave.unsqueeze(0)
                    
                    if processed_frames == 0:
                        if is_last_chunk:
                            output_wave = vc_wave[0].cpu().numpy()
                            generated_wave_chunks.append(output_wave)
                            break
                        output_wave = vc_wave[0, :-self.overlap_wave_len].cpu().numpy()
                        generated_wave_chunks.append(output_wave)
                        previous_chunk = vc_wave[0, -self.overlap_wave_len:]
                        processed_frames += vc_target.size(2) - self.overlap_frame_len
                    elif is_last_chunk:
                        output_wave = self._crossfade(
                            previous_chunk.cpu().numpy(), 
                            vc_wave[0].cpu().numpy(), 
                            self.overlap_wave_len
                        )
                        generated_wave_chunks.append(output_wave)
                        processed_frames += vc_target.size(2) - self.overlap_frame_len
                    else:
                        output_wave = self._crossfade(
                            previous_chunk.cpu().numpy(),
                            vc_wave[0, :-self.overlap_wave_len].cpu().numpy(),
                            self.overlap_wave_len
                        )
                        generated_wave_chunks.append(output_wave)
                        previous_chunk = vc_wave[0, -self.overlap_wave_len:]
                        processed_frames += vc_target.size(2) - self.overlap_frame_len
            
            converted = np.concatenate(generated_wave_chunks)
            
            # Normalize output
            peak = np.abs(converted).max()
            if peak > 0:
                converted = converted * (0.95 / peak)
            
            return converted
        
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: return original audio (resampled if needed)
            if isinstance(audio, np.ndarray):
                return self._resample_audio(
                    audio,
                    self.config.F5_SAMPLE_RATE,
                    self.sr
                )
            return audio
    
    def _crossfade(self, chunk1, chunk2, overlap):
        """Apply crossfade between two audio chunks"""
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
        if len(chunk2) < overlap:
            chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
        else:
            chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
        return chunk2
    
    def convert_all_segments(self, tts_results, reference_audio_path):
        """Apply voice conversion to all segments using Seed-VC"""
        print(f"\n{'='*70}")
        print(f"STEP 5: VOICE CONVERSION (SEED-VC)")
        print(f"{'='*70}")
        
        # Load reference audio
        self.load_reference_audio(reference_audio_path)
        
        print(f"Converting {len(tts_results)} segments with Seed-VC...")
        
        results = []
        total_start = time.time()
        
        for i, result in enumerate(tts_results):
            converted_audio = self.convert_segment(
                audio=result['audio'],
                segment_id=result['segment_id']
            )
            
            # Update result with converted audio
            result['audio'] = converted_audio
            result['sample_rate'] = self.config.SEED_VC_SAMPLE_RATE  # Update sample rate
            results.append(result)
        
        total_elapsed = time.time() - total_start
        print(f"Total time: {total_elapsed:.2f}s ({total_elapsed/len(tts_results):.2f}s per segment avg)")
        
        print(f"Voice conversion complete: {len(results)} segments")
        
        return results

# ============================================================================
# STEP 7: AUDIO ASSEMBLY
# ============================================================================

