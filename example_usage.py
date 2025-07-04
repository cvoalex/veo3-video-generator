#!/usr/bin/env python3
"""
Example usage of the Veo3 Video Generation API
"""

import requests
import time
import json

# API base URL
BASE_URL = "http://localhost:8000"

def generate_simple_video():
    """Generate a simple video with text prompt"""
    print("Generating simple video...")
    
    response = requests.post(
        f"{BASE_URL}/generate",
        data={
            "prompt": "A beautiful sunset over a calm ocean with gentle waves",
            "duration": 8,
            "aspect_ratio": "16:9",
            "external_upload": "false"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Job created: {result['job_id']}")
        return result['job_id']
    else:
        print(f"Error: {response.text}")
        return None

def generate_video_with_audio_and_references():
    """Generate video with audio and reference images"""
    print("\nGenerating video with audio and references...")
    
    # Example with audio file
    with open("ocean_sounds.mp3", "rb") as audio_file:
        files = {"audio_file": ("ocean_sounds.mp3", audio_file, "audio/mpeg")}
        data = {
            "prompt": "A serene beach at sunset with waves matching the audio rhythm",
            "duration": 8,
            "aspect_ratio": "16:9",
            "reference_urls": "https://example.com/beach.jpg,https://example.com/sunset.jpg",
            "style_reference_url": "https://example.com/artistic-style.jpg",
            "external_upload": "true"
        }
        
        response = requests.post(f"{BASE_URL}/generate", data=data, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Job created: {result['job_id']}")
        return result['job_id']
    else:
        print(f"Error: {response.text}")
        return None

def generate_video_chain():
    """Generate a longer video using chaining"""
    print("\nGenerating video chain (32 seconds)...")
    
    response = requests.post(
        f"{BASE_URL}/generate-chain",
        data={
            "base_prompt": "A robot wakes up in a futuristic city at dawn",
            "continuation_prompts": "The robot explores the empty streets,The robot discovers other awakening robots,The robots gather in the city center",
            "total_segments": 4,
            "duration_per_segment": 8,
            "transition_type": "multi_frame",
            "overlap_frames": 3,
            "external_upload": "false"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Chain created: {result['chain_id']}")
        print(f"Total segments: {result['total_segments']}")
        return result['chain_id']
    else:
        print(f"Error: {response.text}")
        return None

def extend_existing_video():
    """Extend an existing video"""
    print("\nExtending existing video...")
    
    response = requests.post(
        f"{BASE_URL}/extend-video",
        data={
            "video_url": "https://storage.googleapis.com/example-bucket/my-video.mp4",
            "continuation_prompt": "The story continues with a dramatic twist as the sun sets",
            "duration": 8,
            "transition_type": "last_frame",
            "overlap_frames": 2
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Extension job created: {result['job_id']}")
        return result['job_id']
    else:
        print(f"Error: {response.text}")
        return None

def check_job_status(job_id):
    """Check the status of a generation job"""
    response = requests.get(f"{BASE_URL}/status/{job_id}")
    
    if response.status_code == 200:
        status = response.json()
        print(f"\nJob {job_id} status:")
        print(f"  Status: {status['status']}")
        print(f"  Progress: {status.get('progress', 0)}%")
        if status.get('video_url'):
            print(f"  Video URL: {status['video_url']}")
        if status.get('error_message'):
            print(f"  Error: {status['error_message']}")
        return status
    else:
        print(f"Error checking status: {response.text}")
        return None

def check_chain_status(chain_id):
    """Check the status of a video chain"""
    response = requests.get(f"{BASE_URL}/chain-status/{chain_id}")
    
    if response.status_code == 200:
        status = response.json()
        print(f"\nChain {chain_id} status:")
        print(f"  Status: {status['status']}")
        print(f"  Progress: {status.get('progress', 0)}%")
        print(f"  Completed segments: {status.get('completed_segments', 0)}")
        if status.get('final_video_url'):
            print(f"  Final video URL: {status['final_video_url']}")
        return status
    else:
        print(f"Error checking chain status: {response.text}")
        return None

def wait_for_completion(job_id, job_type="job", max_wait=600):
    """Wait for a job or chain to complete"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        if job_type == "job":
            status = check_job_status(job_id)
            if status and status['status'] in ['completed', 'failed']:
                return status
        else:
            status = check_chain_status(job_id)
            if status and status['status'] in ['completed', 'failed']:
                return status
        
        time.sleep(5)  # Check every 5 seconds
    
    print(f"Timeout waiting for {job_type} {job_id}")
    return None

def main():
    """Main example flow"""
    print("Veo3 Video Generation API Examples")
    print("==================================")
    
    # Check API health
    try:
        response = requests.get(f"{BASE_URL}/")
        health = response.json()
        print(f"API Status: {health['status']}")
        print(f"Veo3 Available: {health['veo3_available']}")
        
        if not health['veo3_available']:
            print("\nWarning: Veo3 is not available. Please configure Google Cloud credentials.")
            print("The API will run in simulation mode.")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure it's running on http://localhost:8000")
        return
    
    # Example 1: Simple video generation
    job_id = generate_simple_video()
    if job_id:
        print("Waiting for completion...")
        final_status = wait_for_completion(job_id, "job", max_wait=60)
        if final_status and final_status['status'] == 'completed':
            print(f"Video generated successfully: {final_status['video_url']}")
    
    # Example 2: Video chain for longer content
    chain_id = generate_video_chain()
    if chain_id:
        print("Waiting for chain completion...")
        final_status = wait_for_completion(chain_id, "chain", max_wait=300)
        if final_status and final_status['status'] == 'completed':
            print(f"Video chain generated successfully: {final_status['final_video_url']}")
    
    # List all jobs
    print("\nListing all jobs...")
    response = requests.get(f"{BASE_URL}/jobs")
    if response.status_code == 200:
        jobs = response.json()
        print(f"Total jobs: {jobs['total']}")
        for job in jobs['jobs'][:5]:  # Show first 5
            print(f"  - {job['job_id']}: {job['status']} ({job.get('progress', 0)}%)")

if __name__ == "__main__":
    main()