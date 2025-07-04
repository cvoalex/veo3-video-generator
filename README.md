# Veo 3 FastAPI Video Generation Application

A FastAPI application for generating videos using Google's Veo 3 model with support for audio, reference images, video chaining, and external API uploads.

## Features

### Core Features
- ✅ Text-to-video generation using Google Veo 3
- ✅ Audio file support (mp3, wav) for video generation
- ✅ Reference image support via URLs
- ✅ Style transfer with reference images
- ✅ Custom aspect ratios (16:9, 9:16)
- ✅ Seed control for reproducible results
- ✅ Background job processing with progress tracking
- ✅ External API upload integration

### Advanced Features
- ✅ **Video Chaining** - Create longer videos by chaining multiple 8-second segments
- ✅ **Video Extension** - Extend existing videos with new content
- ✅ **Frame Extraction** - Extract frames for continuity between segments
- ✅ **Smart Transitions** - Multiple transition types between video segments
- ✅ **Overlap Handling** - Remove duplicate frames between concatenated videos

## Prerequisites

- Python 3.8+
- Google Cloud Project with Veo 3 API access
- Google Cloud Service Account credentials
- FFmpeg installed on system
- (Optional) External API endpoint for uploads

## Quick Start

### 1. Setup Virtual Environment

```bash
chmod +x setup_venv.sh
./setup_venv.sh
source veo3_env/bin/activate
```

### 2. Configure Environment Variables

Create a `.env` file with your credentials:

```env
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
VEO_LOCATION=us-central1

# External API Configuration (Optional)
EXTERNAL_API_URL=https://your-api.com/upload
EXTERNAL_API_KEY=your-api-key

# App Configuration
DEBUG=True
MAX_FILE_SIZE=100000000  # 100MB
```

### 3. Setup Google Cloud

1. Create a Google Cloud Project
2. Enable the Vertex AI API
3. Create a service account with necessary permissions
4. Download the service account key JSON file
5. Set the path in `.env` file

### 4. Run the Application

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Basic Video Generation

```bash
POST /generate
```

Generate a video with text prompt and optional parameters:

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "prompt=A serene beach at sunset with waves" \
  -F "duration=8" \
  -F "aspect_ratio=16:9" \
  -F "audio_file=@ocean_sounds.mp3" \
  -F "reference_urls=https://example.com/beach.jpg,https://example.com/sunset.jpg" \
  -F "external_upload=true"
```

### Video Chaining (Longer Videos)

```bash
POST /generate-chain
```

Create longer videos by chaining multiple segments:

```bash
curl -X POST "http://localhost:8000/generate-chain" \
  -F "base_prompt=A robot wakes up in a futuristic city" \
  -F "continuation_prompts=The robot explores the city streets,The robot discovers other robots,The robots work together" \
  -F "total_segments=4" \
  -F "duration_per_segment=8" \
  -F "transition_type=multi_frame" \
  -F "overlap_frames=3"
```

This creates a 32-second video (4 segments × 8 seconds) with smooth transitions.

### Video Extension

```bash
POST /extend-video
```

Extend an existing video with new content:

```bash
curl -X POST "http://localhost:8000/extend-video" \
  -F "video_url=https://storage.googleapis.com/my-video.mp4" \
  -F "continuation_prompt=The story continues with a dramatic twist" \
  -F "duration=8" \
  -F "transition_type=last_frame" \
  -F "audio_file=@dramatic_music.mp3"
```

### Status Monitoring

```bash
GET /status/{job_id}
GET /chain-status/{chain_id}
```

Check generation progress:

```bash
curl "http://localhost:8000/status/abc-123-def"
```

Response:
```json
{
  "job_id": "abc-123-def",
  "status": "processing",
  "progress": 75,
  "video_url": null,
  "error_message": null
}
```

### List All Jobs/Chains

```bash
GET /jobs
GET /chains
```

## Video Chaining Details

### Transition Types

1. **last_frame** - Uses the final frame of previous segment as reference
2. **multi_frame** - Uses last 3 frames for better continuity
3. **first_last** - Ensures last frame of segment A matches first frame of segment B

### Overlap Frames

- Set `overlap_frames` to remove duplicate frames during concatenation
- Typically 2-5 frames work best
- Prevents jarring jumps between segments

### Example: 40-Second Story Video

```python
# Create a 40-second video with 5 segments
prompts = [
    "continuation_prompts": [
        "The hero begins their journey",
        "Obstacles appear on the path", 
        "A mentor provides guidance",
        "The final challenge awaits",
        "Victory and celebration"
    ]
]
```

## Audio Integration

The API supports audio files in multiple formats:
- MP3
- WAV
- M4A
- AAC

Audio is synchronized with the generated video content.

## Reference Images

### Style Reference
Use a style reference image to influence the visual style:

```bash
-F "style_reference_url=https://example.com/art-style.jpg"
```

### Content References
Provide multiple reference images for content consistency:

```bash
-F "reference_urls=https://example.com/character.jpg,https://example.com/scene.jpg"
```

## Error Handling

The API includes comprehensive error handling:

- **400** - Invalid request parameters
- **404** - Job/Chain not found
- **413** - File too large
- **500** - Server errors
- **503** - Veo 3 service unavailable

## Production Considerations

1. **Storage**: Configure Google Cloud Storage for video outputs
2. **Authentication**: Implement proper API authentication
3. **Rate Limiting**: Add rate limiting for API endpoints
4. **Job Persistence**: Use Redis or database instead of in-memory storage
5. **Monitoring**: Set up logging and monitoring
6. **Scaling**: Use job queues (Celery, RQ) for better scaling

## Docker Deployment

Build and run with Docker:

```bash
docker build -t veo3-api .
docker run -p 8000:8000 \
  -v ./credentials.json:/app/credentials.json:ro \
  --env-file .env \
  veo3-api
```

## API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

MIT License

## Support

For issues and questions:
- Check the `/docs` endpoint for API documentation
- Review error messages in job status
- Enable DEBUG mode for detailed logging