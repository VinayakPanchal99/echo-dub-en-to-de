# Echo Dub: English to German Video Translation Platform

An AI-powered video dubbing platform that translates English videos to German using state-of-the-art text-to-speech (F5-TTS) and voice conversion (Seed-VC) technologies.

## Overview

This platform automates the complete video dubbing pipeline:
- **Extracts** audio from input video
- **Translates** English subtitles to German using NLLB-200
- **Synthesizes** German speech using F5-TTS
- **Converts** voice to match the original speaker using Seed-VC
- **Assembles** and merges the dubbed audio with the original video
- **Evaluates** quality and generates reports

## Architecture

### Pipeline Flow

```
Input Video (MP4) + SRT Subtitles
    ↓
[1] Audio Extraction → Extract audio track & reference speaker segment
    ↓
[2] SRT Parsing → Parse subtitle segments with timestamps
    ↓
[3] Translation → Translate English subtitles to German (NLLB-200)
    ↓
[4] TTS Synthesis → Generate German speech audio (F5-TTS)
    ↓
[5] Voice Conversion → Convert voice to match original speaker (Seed-VC)
    ↓
[6] Audio Assembly → Assemble all segments with proper timing
    ↓
[7] Video Merging → Merge dubbed audio with original video
    ↓
[8] Quality Evaluation → Generate evaluation report
    ↓
Output Video (MP4) + Evaluation Report
```

### Source Code Structure

```
src/
├── api/
│   └── main.py              # FastAPI REST API endpoints
├── core/
│   └── config.py            # Configuration management
├── pipelines/
│   ├── pipeline.py           # Main pipeline orchestrator
│   ├── audio_extractor.py   # Extract audio from video & detect speech
│   ├── srt_parser.py        # Parse SRT subtitle files
│   ├── translator.py        # NLLB-200 translation engine
│   ├── tts_synthesizer.py   # F5-TTS text-to-speech synthesis
│   ├── voice_converter.py   # Seed-VC voice conversion
│   ├── audio_assembler.py   # Assemble audio segments
│   ├── video_merger.py      # Merge audio with video
│   ├── evaluator.py         # Quality evaluation & reporting
│   └── prosody_analyzer.py  # Prosody analysis (optional)
└── utils/
    ├── device.py            # Device detection (CPU/GPU/MPS)
    └── paths.py             # Path setup utilities
```

### Key Components

#### 1. **Audio Extractor** (`audio_extractor.py`)
- Extracts audio track from video using FFmpeg
- Intelligently detects best speech segment for voice cloning
- Extracts reference audio (5-10 seconds) for voice conversion

#### 2. **SRT Parser** (`srt_parser.py`)
- Parses SRT subtitle files
- Extracts text segments with timestamps
- Handles subtitle formatting and timing

#### 3. **Translator** (`translator.py`)
- Uses Facebook's NLLB-200 (600M) model
- Translates English (`eng_Latn`) to German (`deu_Latn`)
- Supports batch translation with multi-threading

#### 4. **TTS Synthesizer** (`tts_synthesizer.py`)
- Uses F5-TTS v1 Base model for high-quality speech synthesis
- Generates German speech from translated text
- Configurable quality settings (NFE steps, CFG strength)

#### 5. **Voice Converter** (`voice_converter.py`)
- Uses Seed-VC for voice cloning
- Converts synthetic voice to match original speaker characteristics
- Uses DiT (Diffusion Transformer) architecture with Whisper features

#### 6. **Audio Assembler** (`audio_assembler.py`)
- Assembles individual audio segments into complete track
- Handles timing synchronization
- Applies cross-fading for smooth transitions

#### 7. **Video Merger** (`video_merger.py`)
- Merges dubbed audio with original video
- Preserves video quality and metadata
- Uses FFmpeg for final video generation

#### 8. **Quality Evaluator** (`evaluator.py`)
- Evaluates translation accuracy
- Assesses voice similarity and quality
- Generates detailed evaluation reports

## Setup Instructions

### Prerequisites

- Python 3.11
- FFmpeg (for audio/video processing)
- CUDA-capable GPU (optional, for faster processing)

### Step 1: Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### Step 2: Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from [FFmpeg website](https://ffmpeg.org/download.html)

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Setup External Dependencies

```bash
# Create necessary directories
mkdir -p external inputs outputs temp uploads

# Clone and install F5-TTS (Text-to-Speech)
git clone https://github.com/SWivid/F5-TTS.git external/F5-TTS
cd external/F5-TTS
pip install -e .
cd ../..

# Clone and install Seed-VC (Voice Conversion)
git clone https://github.com/Plachtaa/seed-vc.git external/seed-vc
cd external/seed-vc
pip install -r requirements.txt

# For macOS, also install Mac-specific requirements
pip install -r requirements-mac.txt

cd ../..
```

### Step 5: Prepare Input Files

Place your video and subtitle files in the `inputs/` directory:
- Video file: `inputs/your_video.mp4` (MP4 format)
- Subtitle file: `inputs/your_subtitles.srt` (SRT format)

## Usage

### Start the API Server

```bash
python -m src.api.main
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Upload Video and SRT Files

```bash
curl -X POST "http://localhost:8000/dub" \
  -F "video_file=@inputs/Tanzania-2.mp4" \
  -F "srt_file=@inputs/Tanzania-caption.srt"
```

**Response:**
```json
{
  "job_id": "31cd8bed-ae17-4424-b985-f047b1d193cc",
  "status": "queued",
  "message": "Job queued for processing",
  "output_video": null,
  "output_audio": null,
  "evaluation_report": null
}
```

#### 2. Check Job Status

```bash
curl "http://localhost:8000/status/{job_id}"
```

**Response:**
```json
{
  "job_id": "31cd8bed-ae17-4424-b985-f047b1d193cc",
  "status": "completed",
  "message": "Pipeline completed successfully",
  "output_video": "/app/outputs/31cd8bed-ae17-4424-b985-f047b1d193cc/31cd8bed-ae17-4424-b985-f047b1d193cc_dubbed.mp4",
  "output_audio": "/app/outputs/31cd8bed-ae17-4424-b985-f047b1d193cc/31cd8bed-ae17-4424-b985-f047b1d193cc_dubbed_audio.wav",
  "evaluation_report": "/app/outputs/31cd8bed-ae17-4424-b985-f047b1d193cc/31cd8bed-ae17-4424-b985-f047b1d193cc_evaluation_report.md"
}
```

**Status values:**
- `queued`: Job is waiting to be processed
- `processing`: Pipeline is running
- `completed`: Job finished successfully
- `failed`: Job encountered an error

#### 3. Download Dubbed Video

```bash
curl "http://localhost:8000/download/{job_id}/video" -o output.mp4
```

#### 4. Download Dubbed Audio

```bash
curl "http://localhost:8000/download/{job_id}/audio" -o dubbed_audio.wav
```

#### 5. Download Evaluation Report

```bash
curl "http://localhost:8000/download/{job_id}/report" -o evaluation_report.md
```

#### 6. Delete Job

```bash
curl -X DELETE "http://localhost:8000/job/{job_id}"
```

### Health Check

```bash
curl "http://localhost:8000/"
```

## Output Structure

```
outputs/
└── {job_id}/
    ├── {job_id}_dubbed.mp4              # Final dubbed video
    ├── {job_id}_dubbed_audio.wav        # Dubbed audio file
    └── {job_id}_evaluation_report.md    # Quality evaluation report
```

## Technology Stack

- **FastAPI**: REST API framework
- **F5-TTS**: Text-to-speech synthesis
- **Seed-VC**: Voice conversion with Diffusion Transformers
- **NLLB-200**: Multi-lingual translation model
- **FFmpeg**: Audio/video processing
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers library

## Notes and Limitations

1. Pip dependency error: This error can be ignored since the descript-audio tools is not being used.
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
descript-audiotools 0.7.2 requires protobuf<3.20,>=3.9.2, but you have protobuf 3.20.3 which is incompatible.
```

2. Language Restriction: Currently limited to English-to-German translation only. The system is not designed for other language pairs. 

3. Hardware Dependency: While CPU processing is supported, optimal performance requires a CUDA-capable GPU. Processing times may be significantly longer without GPU acceleration.

4. Sequential Processing: The pipeline operates as a single-threaded job queue with sequential stages, potentially creating bottlenecks for batch processing multiple videos simultaneously.

5. Dependency Complexity: Requires external cloning and installation of F5-TTS and Seed-VC repositories, plus platform-specific requirements (Mac-specific for macOS), adding setup complexity

6. Lip-Sync integration: Lip-sync is not integrated in this pipeline but model like `WaV2lip` can be added to the pipeline.

## License

See individual component licenses (can be viewed after cloning these repo):
- F5-TTS: See `external/F5-TTS/LICENSE`
- Seed-VC: See `external/seed-vc/LICENSE`