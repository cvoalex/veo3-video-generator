import os
import asyncio
import tempfile
import base64
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import logging
from datetime import datetime

import cv2
import numpy as np
import ffmpeg
from PIL import Image
import httpx
import aiofiles
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

from google.cloud import aiplatform
from google.auth.exceptions import DefaultCredentialsError
import google.auth
from google.auth.transport import requests as auth_requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Veo 3 Video Generator API",
    description="Generate videos using Google Veo 3 and upload to external APIs",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
LOCATION = os.getenv("VEO_LOCATION", "us-central1")
EXTERNAL_API_URL = os.getenv("EXTERNAL_API_URL")
EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "100000000"))

# Models
class VideoGenerationRequest(BaseModel):
    prompt: str
    duration: Optional[int] = 8  # seconds
    aspect_ratio: Optional[str] = "16:9"  # or "9:16"
    seed: Optional[int] = None
    reference_urls: Optional[List[HttpUrl]] = None
    style_reference_url: Optional[HttpUrl] = None
    external_upload: Optional[bool] = True
    audio_data: Optional[str] = None  # Base64 encoded audio
    audio_format: Optional[str] = None  # mp3, wav, etc.

class VideoChainRequest(BaseModel):
    base_prompt: str
    continuation_prompts: List[str]
    total_segments: int
    duration_per_segment: Optional[int] = 8
    aspect_ratio: Optional[str] = "16:9"
    overlap_frames: Optional[int] = 3
    transition_type: Optional[str] = "last_frame"  # "last_frame", "multi_frame", "first_last"
    external_upload: Optional[bool] = True

class VideoGenerationResponse(BaseModel):
    job_id: str
    status: str
    message: str
    video_url: Optional[str] = None
    external_upload_result: Optional[dict] = None

class VideoChainResponse(BaseModel):
    chain_id: str
    status: str
    message: str
    total_segments: int
    completed_segments: int
    segment_urls: List[str] = []
    final_video_url: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[int] = None
    video_url: Optional[str] = None
    error_message: Optional[str] = None

# In-memory job storage (use Redis/database in production)
jobs = {}
video_chains = {}

class VideoFrameExtractor:
    """Utility class for extracting frames from videos"""
    
    @staticmethod
    async def download_video(url: str) -> str:
        """Download video from URL and return local path"""
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(response.content)
            temp_file.close()
            
            return temp_file.name
    
    @staticmethod
    def extract_last_frames(video_path: str, num_frames: int = 3) -> List[str]:
        """Extract last N frames from video and return as base64 encoded images"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate which frames to extract
        start_frame = max(0, total_frames - num_frames)
        frame_indices = range(start_frame, total_frames)
        
        extracted_frames = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                pil_image.save(temp_file.name, 'JPEG', quality=95)
                
                extracted_frames.append(temp_file.name)
        
        cap.release()
        return extracted_frames
    
    @staticmethod
    def extract_frame_at_timestamp(video_path: str, timestamp: float) -> str:
        """Extract frame at specific timestamp"""
        cap = cv2.VideoCapture(video_path)
        
        # Seek to timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            pil_image.save(temp_file.name, 'JPEG', quality=95)
            
            cap.release()
            return temp_file.name
        
        cap.release()
        return None
    
    @staticmethod
    async def concatenate_videos(video_paths: List[str], output_path: str, overlap_frames: int = 0):
        """Concatenate multiple videos with optional frame overlap removal"""
        try:
            if overlap_frames > 0:
                # Process videos to remove overlapping frames
                processed_paths = []
                
                for i, video_path in enumerate(video_paths):
                    if i == 0:
                        # First video - keep as is
                        processed_paths.append(video_path)
                    else:
                        # Remove first N frames from subsequent videos
                        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                        
                        # Get video info
                        probe = ffmpeg.probe(video_path)
                        fps = eval(probe['streams'][0]['r_frame_rate'])
                        
                        # Calculate time to skip
                        skip_time = overlap_frames / fps
                        
                        # Trim video
                        (
                            ffmpeg
                            .input(video_path, ss=skip_time)
                            .output(temp_output, vcodec='libx264', acodec='aac')
                            .overwrite_output()
                            .run(quiet=True)
                        )
                        
                        processed_paths.append(temp_output)
                
                video_paths = processed_paths
            
            # Create concat file
            concat_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
            for video_path in video_paths:
                concat_file.write(f"file '{video_path}'\n")
            concat_file.close()
            
            # Concatenate videos
            (
                ffmpeg
                .input(concat_file.name, format='concat', safe=0)
                .output(output_path, vcodec='libx264', acodec='aac')
                .overwrite_output()
                .run(quiet=True)
            )
            
            # Cleanup
            os.unlink(concat_file.name)
            if overlap_frames > 0:
                for path in video_paths[1:]:  # Don't delete the first original video
                    if os.path.exists(path):
                        os.unlink(path)
                        
        except Exception as e:
            raise Exception(f"Video concatenation failed: {str(e)}")

class Veo3Client:
    def __init__(self):
        try:
            aiplatform.init(project=PROJECT_ID, location=LOCATION)
            self.model_endpoint = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/veo-001-preview-0827"
            
            # Get credentials
            self.credentials, self.project = google.auth.default()
            self.auth_request = auth_requests.Request()
            
        except DefaultCredentialsError:
            raise HTTPException(
                status_code=500,
                detail="Google Cloud credentials not configured. Please set GOOGLE_APPLICATION_CREDENTIALS"
            )
    
    async def download_reference_image(self, url: str) -> str:
        """Download reference image from URL and return local path"""
        async with httpx.AsyncClient() as client:
            response = await client.get(str(url))
            response.raise_for_status()
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file.write(response.content)
            temp_file.close()
            
            return temp_file.name
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    async def generate_video(self, request: VideoGenerationRequest, job_id: str):
        """Generate video using Veo 3"""
        try:
            jobs[job_id]["status"] = "processing"
            jobs[job_id]["progress"] = 10
            
            # Refresh credentials if needed
            self.credentials.refresh(self.auth_request)
            
            # Prepare the request payload for Veo3
            generation_config = {
                "prompt": request.prompt,
                "negative_prompt": "",  # Add if needed
                "height": 1080 if request.aspect_ratio == "16:9" else 1920,
                "width": 1920 if request.aspect_ratio == "16:9" else 1080,
                "frame_rate": 24,
                "duration": request.duration,
                "camera_motion": "AUTO",  # Can be customized
                "seed": request.seed if request.seed else None
            }
            
            # Handle reference images
            if request.reference_urls:
                reference_images = []
                for url in request.reference_urls:
                    image_path = await self.download_reference_image(url)
                    base64_image = self.encode_image_to_base64(image_path)
                    reference_images.append({
                        "image_bytes": base64_image,
                        "image_type": "IMAGE"
                    })
                    os.unlink(image_path)  # Clean up
                generation_config["reference_images"] = reference_images
            
            # Handle style reference
            if request.style_reference_url:
                style_path = await self.download_reference_image(request.style_reference_url)
                style_base64 = self.encode_image_to_base64(style_path)
                generation_config["style_reference"] = {
                    "image_bytes": style_base64,
                    "image_type": "STYLE"
                }
                os.unlink(style_path)
            
            # Handle audio if provided
            if request.audio_data:
                generation_config["audio"] = {
                    "audio_bytes": request.audio_data,
                    "audio_format": request.audio_format or "mp3"
                }
            
            jobs[job_id]["progress"] = 30
            
            # Create the generation request
            instances = [generation_config]
            parameters = {
                "output_gcs_uri": f"gs://{PROJECT_ID}-veo3-outputs/{job_id}/",
                "output_format": "mp4"
            }
            
            jobs[job_id]["progress"] = 50
            
            # Make the prediction request using Vertex AI
            endpoint = aiplatform.Endpoint(self.model_endpoint)
            
            # Submit the job
            response = endpoint.predict(
                instances=instances,
                parameters=parameters
            )
            
            # Extract job name from response
            operation_name = response.deployed_model_id  # This might vary based on actual API
            
            jobs[job_id]["operation_name"] = operation_name
            jobs[job_id]["progress"] = 60
            
            # Poll for completion
            await self.poll_operation(operation_name, job_id)
            
            # Get the generated video URL
            video_url = f"gs://{PROJECT_ID}-veo3-outputs/{job_id}/output.mp4"
            jobs[job_id]["video_url"] = video_url
            jobs[job_id]["progress"] = 90
            
            # Upload to external API if requested
            if request.external_upload and EXTERNAL_API_URL:
                upload_result = await self.upload_to_external_api(video_url, job_id)
                jobs[job_id]["external_upload_result"] = upload_result
            
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            
        except Exception as e:
            logger.error(f"Video generation failed for job {job_id}: {str(e)}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error_message"] = str(e)
    
    async def poll_operation(self, operation_name: str, job_id: str, max_wait: int = 600):
        """Poll for operation completion"""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < max_wait:
            try:
                # Check operation status
                # In actual implementation, use the appropriate Google Cloud API
                # This is a placeholder
                await asyncio.sleep(5)
                
                # Update progress
                elapsed = asyncio.get_event_loop().time() - start_time
                progress = min(60 + (elapsed / max_wait) * 30, 89)
                jobs[job_id]["progress"] = int(progress)
                
                # Check if complete (placeholder logic)
                if elapsed > 30:  # Simulate completion after 30 seconds
                    return
                    
            except Exception as e:
                logger.error(f"Error polling operation: {str(e)}")
                raise
        
        raise TimeoutError("Operation timed out")
    
    async def upload_to_external_api(self, video_url: str, job_id: str) -> dict:
        """Upload video to external API"""
        try:
            # For GCS URLs, we need to download first or use signed URLs
            # This is a simplified version
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                # Create a signed URL for the GCS object
                # In production, implement proper GCS signed URL generation
                
                # For now, simulate with a placeholder
                download_url = video_url  # This would be a signed URL in production
                
                # Download video
                video_response = await client.get(download_url)
                video_response.raise_for_status()
                
                # Upload to external API
                files = {"video": (f"{job_id}.mp4", video_response.content, "video/mp4")}
                headers = {"Authorization": f"Bearer {EXTERNAL_API_KEY}"}
                
                upload_response = await client.post(
                    EXTERNAL_API_URL,
                    files=files,
                    headers=headers
                )
                upload_response.raise_for_status()
                
                return upload_response.json()
                
        except Exception as e:
            logger.error(f"External API upload failed: {str(e)}")
            return {"error": str(e)}
    
    async def generate_video_chain(self, request: VideoChainRequest, chain_id: str):
        """Generate a chain of connected videos for longer sequences"""
        try:
            video_chains[chain_id]["status"] = "processing"
            video_chains[chain_id]["progress"] = 5
            
            segment_urls = []
            previous_video_path = None
            reference_images = []
            
            # Generate each segment
            for segment_idx in range(request.total_segments):
                segment_progress = (segment_idx / request.total_segments) * 80
                video_chains[chain_id]["progress"] = 5 + segment_progress
                
                # Determine prompt for this segment
                if segment_idx == 0:
                    current_prompt = request.base_prompt
                else:
                    prompt_idx = min(segment_idx - 1, len(request.continuation_prompts) - 1)
                    current_prompt = request.continuation_prompts[prompt_idx]
                
                # Create video generation request
                segment_request = VideoGenerationRequest(
                    prompt=current_prompt,
                    duration=request.duration_per_segment,
                    aspect_ratio=request.aspect_ratio,
                    reference_urls=None,  # Will be set below
                    external_upload=False  # Handle upload at the end
                )
                
                # Add reference images from previous video
                if previous_video_path and reference_images:
                    # Upload reference images to temporary URLs (in production, use cloud storage)
                    reference_urls = []
                    for img_path in reference_images:
                        # In a real implementation, upload to cloud storage and get URLs
                        # For now, we'll simulate this
                        temp_url = f"file://{img_path}"
                        reference_urls.append(temp_url)
                    
                    segment_request.reference_urls = reference_urls
                
                # Generate segment
                segment_job_id = f"{chain_id}_segment_{segment_idx}"
                jobs[segment_job_id] = {
                    "status": "processing",
                    "progress": 0,
                    "video_url": None,
                    "error_message": None
                }
                
                await self.generate_video(segment_request, segment_job_id)
                
                # Check if generation was successful
                if jobs[segment_job_id]["status"] != "completed":
                    raise Exception(f"Segment {segment_idx} generation failed: {jobs[segment_job_id].get('error_message', 'Unknown error')}")
                
                segment_url = jobs[segment_job_id]["video_url"]
                segment_urls.append(segment_url)
                
                # Prepare for next iteration
                if segment_idx < request.total_segments - 1:
                    # Download current video
                    current_video_path = await VideoFrameExtractor.download_video(segment_url)
                    
                    # Extract frames based on transition type
                    if request.transition_type == "last_frame":
                        reference_images = VideoFrameExtractor.extract_last_frames(current_video_path, 1)
                    elif request.transition_type == "multi_frame":
                        reference_images = VideoFrameExtractor.extract_last_frames(current_video_path, 3)
                    elif request.transition_type == "first_last":
                        # Extract last frame for next video's first frame
                        last_frame = VideoFrameExtractor.extract_last_frames(current_video_path, 1)
                        reference_images = last_frame
                    
                    previous_video_path = current_video_path
                
                # Update chain status
                video_chains[chain_id]["completed_segments"] = segment_idx + 1
                video_chains[chain_id]["segment_urls"] = segment_urls
            
            # Concatenate all segments into final video
            video_chains[chain_id]["progress"] = 90
            
            # Download all segment videos
            local_video_paths = []
            for url in segment_urls:
                local_path = await VideoFrameExtractor.download_video(url)
                local_video_paths.append(local_path)
            
            # Create final concatenated video
            final_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            await VideoFrameExtractor.concatenate_videos(
                local_video_paths, 
                final_video_path, 
                overlap_frames=request.overlap_frames
            )
            
            # Upload final video (simulate cloud storage upload)
            final_video_url = f"gs://{PROJECT_ID}-veo3-outputs/chains/{chain_id}_final.mp4"
            video_chains[chain_id]["final_video_url"] = final_video_url
            
            # Upload to external API if requested
            if request.external_upload and EXTERNAL_API_URL:
                upload_result = await self.upload_to_external_api(final_video_url, chain_id)
                video_chains[chain_id]["external_upload_result"] = upload_result
            
            video_chains[chain_id]["status"] = "completed"
            video_chains[chain_id]["progress"] = 100
            
            # Cleanup temporary files
            for path in local_video_paths:
                if os.path.exists(path):
                    os.unlink(path)
            
            # Cleanup reference images
            for img_path in reference_images:
                if os.path.exists(img_path):
                    os.unlink(img_path)
                    
        except Exception as e:
            logger.error(f"Video chain generation failed: {str(e)}")
            video_chains[chain_id]["status"] = "failed"
            video_chains[chain_id]["error_message"] = str(e)

# Initialize Veo 3 client
try:
    veo3_client = Veo3Client()
except HTTPException as e:
    veo3_client = None
    logger.warning(f"Veo3Client initialization failed: {e.detail}")

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    if not veo3_client:
        logger.warning("Running without Veo 3 integration. Please configure Google Cloud credentials.")
    else:
        logger.info("Veo 3 client initialized successfully")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Veo 3 Video Generator API",
        "status": "running",
        "veo3_available": veo3_client is not None,
        "version": "1.0.0"
    }

@app.post("/generate", response_model=VideoGenerationResponse)
async def generate_video(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    duration: Optional[int] = Form(8),
    aspect_ratio: Optional[str] = Form("16:9"),
    seed: Optional[int] = Form(None),
    reference_urls: Optional[str] = Form(None),  # Comma-separated URLs
    style_reference_url: Optional[str] = Form(None),
    external_upload: Optional[bool] = Form(True),
    audio_file: Optional[UploadFile] = File(None)
):
    """Generate video using Veo 3"""
    
    if not veo3_client:
        raise HTTPException(
            status_code=503,
            detail="Veo 3 service not available. Please configure Google Cloud credentials."
        )
    
    # Validate file size
    if audio_file and audio_file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large. Maximum size: {MAX_FILE_SIZE} bytes"
        )
    
    # Parse reference URLs
    parsed_reference_urls = None
    if reference_urls:
        try:
            parsed_reference_urls = [HttpUrl(url.strip()) for url in reference_urls.split(',')]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid reference URLs: {str(e)}")
    
    # Parse style reference URL
    parsed_style_url = None
    if style_reference_url:
        try:
            parsed_style_url = HttpUrl(style_reference_url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid style reference URL: {str(e)}")
    
    # Handle audio file if provided
    audio_data = None
    audio_format = None
    if audio_file:
        audio_content = await audio_file.read()
        audio_data = base64.b64encode(audio_content).decode('utf-8')
        audio_format = audio_file.filename.split('.')[-1].lower()
    
    # Create request object
    request_obj = VideoGenerationRequest(
        prompt=prompt,
        duration=duration,
        aspect_ratio=aspect_ratio,
        seed=seed,
        reference_urls=parsed_reference_urls,
        style_reference_url=parsed_style_url,
        external_upload=external_upload,
        audio_data=audio_data,
        audio_format=audio_format
    )
    
    # Generate job ID
    import uuid
    job_id = str(uuid.uuid4())
    
    # Initialize job
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "video_url": None,
        "error_message": None,
        "external_upload_result": None,
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Start video generation in background
    background_tasks.add_task(veo3_client.generate_video, request_obj, job_id)
    
    return VideoGenerationResponse(
        job_id=job_id,
        status="queued",
        message="Video generation started"
    )

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        video_url=job.get("video_url"),
        error_message=job.get("error_message")
    )

@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": job_id,
                "status": job_data["status"],
                "created_at": job_data.get("created_at"),
                "progress": job_data.get("progress")
            }
            for job_id, job_data in jobs.items()
        ]
    }

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del jobs[job_id]
    return {"message": "Job deleted successfully"}

@app.post("/generate-chain", response_model=VideoChainResponse)
async def generate_video_chain(
    background_tasks: BackgroundTasks,
    base_prompt: str = Form(...),
    continuation_prompts: str = Form(...),  # Comma-separated prompts
    total_segments: int = Form(...),
    duration_per_segment: Optional[int] = Form(8),
    aspect_ratio: Optional[str] = Form("16:9"),
    overlap_frames: Optional[int] = Form(3),
    transition_type: Optional[str] = Form("last_frame"),
    external_upload: Optional[bool] = Form(True)
):
    """Generate a chain of connected videos for longer sequences"""
    
    if not veo3_client:
        raise HTTPException(
            status_code=503,
            detail="Veo 3 service not available. Please configure Google Cloud credentials."
        )
    
    # Parse continuation prompts
    try:
        parsed_prompts = [prompt.strip() for prompt in continuation_prompts.split(',')]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid continuation prompts: {str(e)}")
    
    # Validate inputs
    if total_segments < 2:
        raise HTTPException(status_code=400, detail="Total segments must be at least 2")
    
    if transition_type not in ["last_frame", "multi_frame", "first_last"]:
        raise HTTPException(status_code=400, detail="Invalid transition type")
    
    # Create request object
    request_obj = VideoChainRequest(
        base_prompt=base_prompt,
        continuation_prompts=parsed_prompts,
        total_segments=total_segments,
        duration_per_segment=duration_per_segment,
        aspect_ratio=aspect_ratio,
        overlap_frames=overlap_frames,
        transition_type=transition_type,
        external_upload=external_upload
    )
    
    # Generate chain ID
    import uuid
    chain_id = str(uuid.uuid4())
    
    # Initialize chain
    video_chains[chain_id] = {
        "status": "queued",
        "progress": 0,
        "completed_segments": 0,
        "segment_urls": [],
        "final_video_url": None,
        "error_message": None,
        "external_upload_result": None,
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Start video chain generation in background
    background_tasks.add_task(veo3_client.generate_video_chain, request_obj, chain_id)
    
    return VideoChainResponse(
        chain_id=chain_id,
        status="queued",
        message="Video chain generation started",
        total_segments=total_segments,
        completed_segments=0
    )

@app.get("/chain-status/{chain_id}")
async def get_chain_status(chain_id: str):
    """Get video chain status"""
    if chain_id not in video_chains:
        raise HTTPException(status_code=404, detail="Chain not found")
    
    chain = video_chains[chain_id]
    return {
        "chain_id": chain_id,
        "status": chain["status"],
        "progress": chain.get("progress"),
        "completed_segments": chain.get("completed_segments", 0),
        "segment_urls": chain.get("segment_urls", []),
        "final_video_url": chain.get("final_video_url"),
        "error_message": chain.get("error_message"),
        "created_at": chain.get("created_at")
    }

@app.post("/extend-video")
async def extend_video(
    background_tasks: BackgroundTasks,
    video_url: str = Form(...),
    continuation_prompt: str = Form(...),
    duration: Optional[int] = Form(8),
    transition_type: Optional[str] = Form("last_frame"),
    overlap_frames: Optional[int] = Form(2),
    audio_file: Optional[UploadFile] = File(None)
):
    """Extend an existing video by generating a continuation"""
    
    if not veo3_client:
        raise HTTPException(
            status_code=503,
            detail="Veo 3 service not available. Please configure Google Cloud credentials."
        )
    
    try:
        # Download the existing video
        video_path = await VideoFrameExtractor.download_video(video_url)
        
        # Extract reference frames
        if transition_type == "last_frame":
            reference_images = VideoFrameExtractor.extract_last_frames(video_path, 1)
        elif transition_type == "multi_frame":
            reference_images = VideoFrameExtractor.extract_last_frames(video_path, 3)
        else:
            reference_images = VideoFrameExtractor.extract_last_frames(video_path, 1)
        
        # Upload reference images (simulate URLs)
        reference_urls = [f"file://{img}" for img in reference_images]
        
        # Handle audio if provided
        audio_data = None
        audio_format = None
        if audio_file:
            audio_content = await audio_file.read()
            audio_data = base64.b64encode(audio_content).decode('utf-8')
            audio_format = audio_file.filename.split('.')[-1].lower()
        
        # Create generation request
        request_obj = VideoGenerationRequest(
            prompt=continuation_prompt,
            duration=duration,
            reference_urls=reference_urls,
            external_upload=False,
            audio_data=audio_data,
            audio_format=audio_format
        )
        
        # Generate job ID
        import uuid
        job_id = str(uuid.uuid4())
        
        # Initialize job
        jobs[job_id] = {
            "status": "queued",
            "progress": 0,
            "video_url": None,
            "error_message": None,
            "original_video": video_url,
            "is_extension": True,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Start generation
        background_tasks.add_task(veo3_client.generate_video, request_obj, job_id)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Video extension started",
            "original_video": video_url
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extend video: {str(e)}")

@app.get("/chains")
async def list_chains():
    """List all video chains"""
    return {
        "total": len(video_chains),
        "chains": [
            {
                "chain_id": chain_id,
                "status": chain_data["status"],
                "created_at": chain_data.get("created_at"),
                "progress": chain_data.get("progress"),
                "completed_segments": chain_data.get("completed_segments", 0)
            }
            for chain_id, chain_data in video_chains.items()
        ]
    }

@app.delete("/chains/{chain_id}")
async def delete_chain(chain_id: str):
    """Delete a video chain"""
    if chain_id not in video_chains:
        raise HTTPException(status_code=404, detail="Chain not found")
    
    del video_chains[chain_id]
    return {"message": "Chain deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("DEBUG", "False").lower() == "true"
    )