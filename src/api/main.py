"""
FastAPI Backend for Video Dubbing Pipeline
"""
import os
import sys
import uuid
from pathlib import Path
from typing import Optional
import shutil

# Enable MPS fallback for unsupported operations (must be before any torch imports)
if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') is None:
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Add external dependencies to path
F5_TTS_PATH = PROJECT_ROOT / "external" / "F5-TTS" / "src"
if F5_TTS_PATH.exists():
    sys.path.insert(0, str(F5_TTS_PATH))

SEED_VC_PATH = PROJECT_ROOT / "external" / "seed-vc"
if SEED_VC_PATH.exists():
    sys.path.insert(0, str(SEED_VC_PATH))

# Import pipeline from modular structure
from src.core.config import Config
from src.pipelines.pipeline import VideoDubbingPipeline

app = FastAPI(
    title="Video Dubbing API",
    description="API for video dubbing with F5-TTS and Seed-VC",
    version="0.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TEMP_DIR = PROJECT_ROOT / "temp"

for dir_path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class JobStatus(BaseModel):
    job_id: str
    status: str
    message: str
    output_video: Optional[str] = None
    output_audio: Optional[str] = None
    evaluation_report: Optional[str] = None


# In-memory job status storage (use Redis in production)
job_status = {}


def run_dubbing_pipeline(job_id: str, video_path: Path, srt_path: Path, output_dir: Path):
    """Run the dubbing pipeline in background"""
    try:
        job_status[job_id] = {
            "status": "processing",
            "message": "Pipeline started",
            "output_video": None,
            "evaluation_report": None
        }
        
        # Create config for this job
        config = Config()
        config.INPUT_VIDEO = video_path
        config.INPUT_SRT = srt_path
        config.OUTPUT_DIR = output_dir
        config.OUTPUT_VIDEO = output_dir / f"{job_id}_dubbed.mp4"
        config.TEMP_DIR = TEMP_DIR / job_id
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        
        # Update temp file paths
        config.EXTRACTED_AUDIO = config.TEMP_DIR / "original_audio.wav"
        config.REFERENCE_AUDIO = config.TEMP_DIR / "reference_speaker.wav"
        config.TRANSLATED_SRT = config.TEMP_DIR / "subtitles_de.srt"
        config.TTS_AUDIO = config.TEMP_DIR / "tts_audio.wav"
        config.DUBBED_AUDIO = config.TEMP_DIR / "dubbed_audio.wav"
        config.PROSODY_OUTPUT = config.TEMP_DIR / "prosody_analysis.json"
        
        # Run pipeline
        pipeline = VideoDubbingPipeline(config)
        pipeline.run()
        
        # Update status
        eval_report = output_dir / f"{job_id}_evaluation_report.md"
        output_audio = output_dir / f"{job_id}_dubbed_audio.wav"
        
        if config.OUTPUT_VIDEO.exists():
            # Copy dubbed audio to output directory
            if config.DUBBED_AUDIO.exists():
                shutil.copy2(config.DUBBED_AUDIO, output_audio)
            
            # Rename evaluation report if it exists (avoid duplicate files)
            # The evaluator saves to OUTPUT_DIR, check there
            eval_source = config.OUTPUT_DIR / "evaluation_report_seedvc.md"
            if not eval_source.exists():
                eval_source = PROJECT_ROOT / "outputs" / "evaluation_report_seedvc.md"
            if eval_source.exists():
                # Rename/move instead of copy to avoid duplicates
                eval_source.rename(eval_report)
            
            # Print final summary after pipeline completes
            print("\n" + "="*70)
            print("PIPELINE COMPLETE")
            print("="*70)
            print(f"Output Video: {config.OUTPUT_VIDEO}")
            if output_audio.exists():
                print(f"Output Audio: {output_audio}")
            if eval_report.exists():
                print(f"Evaluation Report: {eval_report}")
            print(f"Translated SRT: {config.TRANSLATED_SRT}")
            print("="*70 + "\n")
            
            # Use relative paths for API responses
            output_video_rel = f"/app/outputs/{job_id}/{job_id}_dubbed.mp4"
            output_audio_rel = f"/app/outputs/{job_id}/{job_id}_dubbed_audio.wav" if output_audio.exists() else None
            eval_report_rel = f"/app/outputs/{job_id}/{job_id}_evaluation_report.md" if eval_report.exists() else None
            
            job_status[job_id] = {
                "status": "completed",
                "message": "Pipeline completed successfully",
                "output_video": output_video_rel,
                "output_audio": output_audio_rel,
                "evaluation_report": eval_report_rel
            }
        else:
            job_status[job_id] = {
                "status": "failed",
                "message": "Pipeline completed but output video not found",
                "output_video": None,
                "output_audio": None,
                "evaluation_report": None
            }
    except Exception as e:
        job_status[job_id] = {
            "status": "failed",
            "message": f"Error: {str(e)}",
            "output_video": None,
            "output_audio": None,
            "evaluation_report": None
        }


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Video Dubbing API is running", "version": "0.1.0"}


@app.post("/dub", response_model=JobStatus)
async def dub_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    srt_file: UploadFile = File(...)
):
    """
    Upload video and SRT files for dubbing
    
    - **video_file**: MP4 video file (English)
    - **srt_file**: SRT subtitle file (English)
    
    Returns a job_id to track processing status
    """
    # Validate file types
    if not video_file.filename or not video_file.filename.endswith(('.mp4', '.MP4')):
        raise HTTPException(status_code=400, detail="Video file must be MP4")
    
    if not srt_file.filename or not srt_file.filename.endswith(('.srt', '.SRT')):
        raise HTTPException(status_code=400, detail="Subtitle file must be SRT")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded files
    video_path = job_dir / video_file.filename
    srt_path = job_dir / srt_file.filename
    
    try:
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video_file.file, f)
        
        with open(srt_path, "wb") as f:
            shutil.copyfileobj(srt_file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving files: {str(e)}")
    
    # Create output directory for this job
    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize job status
    job_status[job_id] = {
        "status": "queued",
        "message": "Job queued for processing",
        "output_video": None,
        "output_audio": None,
        "evaluation_report": None
    }
    
    # Start background task
    background_tasks.add_task(run_dubbing_pipeline, job_id, video_path, srt_path, output_dir)
    
    return JobStatus(
        job_id=job_id,
        status="queued",
        message="Job queued for processing",
        output_video=None,
        output_audio=None,
        evaluation_report=None
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Get the status of a dubbing job"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = job_status[job_id]
    return JobStatus(
        job_id=job_id,
        status=status["status"],
        message=status["message"],
        output_video=status.get("output_video"),
        output_audio=status.get("output_audio"),
        evaluation_report=status.get("evaluation_report")
    )


@app.get("/download/{job_id}/video")
async def download_video(job_id: str):
    """Download the dubbed video"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = job_status[job_id]
    if status["status"] != "completed" or not status.get("output_video"):
        raise HTTPException(status_code=404, detail="Video not ready yet")
    
    # Handle both absolute and relative paths
    video_path_str = status["output_video"]
    if video_path_str.startswith("/app/"):
        # Relative path - convert to absolute
        video_path = PROJECT_ROOT / video_path_str.replace("/app/", "")
    else:
        video_path = Path(video_path_str)
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"{job_id}_dubbed.mp4"
    )


@app.get("/download/{job_id}/audio")
async def download_audio(job_id: str):
    """Download the dubbed audio file"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = job_status[job_id]
    if status["status"] != "completed" or not status.get("output_audio"):
        raise HTTPException(status_code=404, detail="Audio not ready yet")
    
    # Handle both absolute and relative paths
    audio_path_str = status["output_audio"]
    if audio_path_str.startswith("/app/"):
        # Relative path - convert to absolute
        audio_path = PROJECT_ROOT / audio_path_str.replace("/app/", "")
    else:
        audio_path = Path(audio_path_str)
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        filename=f"{job_id}_dubbed_audio.wav"
    )


@app.get("/download/{job_id}/report")
async def download_report(job_id: str):
    """Download the evaluation report"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = job_status[job_id]
    if not status.get("evaluation_report"):
        raise HTTPException(status_code=404, detail="Report not available")
    
    # Handle both absolute and relative paths
    report_path_str = status["evaluation_report"]
    if report_path_str.startswith("/app/"):
        # Relative path - convert to absolute
        report_path = PROJECT_ROOT / report_path_str.replace("/app/", "")
    else:
        report_path = Path(report_path_str)
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found")
    
    return FileResponse(
        path=str(report_path),
        media_type="text/markdown",
        filename=f"{job_id}_evaluation_report.md"
    )


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Remove files
    job_dir = UPLOAD_DIR / job_id
    output_dir = OUTPUT_DIR / job_id
    temp_dir = TEMP_DIR / job_id
    
    for dir_path in [job_dir, output_dir, temp_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
    
    # Remove from status
    del job_status[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

