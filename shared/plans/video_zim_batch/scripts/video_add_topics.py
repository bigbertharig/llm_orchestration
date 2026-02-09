#!/usr/bin/env python3
"""
Video Topic Identification - Add LLM-generated topics to a transcribed video.

This script reads an existing transcript (from Whisper step) and uses the
worker's LLM to identify topics and keywords for each segment.

Usage:
    python video_add_topics.py --batch-id abc123 --video-id my_video_id

Uses WORKER_OLLAMA_URL environment variable (set by worker.py).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from filelock import FileLock

# Add lib folder to path
LIB_PATH = Path(__file__).parent.parent / "lib"
sys.path.insert(0, str(LIB_PATH))

from batch_logger import BatchLogger


def identify_topics_with_llm(segments: list, ollama_url: str = "http://localhost:11434") -> list:
    """
    Use Ollama/Qwen to identify topics for each segment.

    Args:
        segments: List of segment dicts with text
        ollama_url: Ollama API URL (from WORKER_OLLAMA_URL)

    Returns:
        Segments with added topic and keywords fields
    """
    import requests

    print(f"[Topics] Identifying topics for {len(segments)} segments using {ollama_url}...")

    for i, segment in enumerate(segments):
        text = segment.get("text", "")
        if not text:
            segment["topic"] = f"Segment {i+1}"
            segment["keywords"] = []
            continue

        prompt = f"""Analyze this transcript segment and provide:
1. A short topic title (3-7 words)
2. Key items/concepts mentioned (comma-separated list)

Transcript ({segment.get('start_sec', 0):.0f}s - {segment.get('end_sec', 0):.0f}s):
"{text[:1500]}"

Respond in this exact format:
TOPIC: <title>
KEYWORDS: <keyword1>, <keyword2>, ...
"""

        try:
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": "qwen2.5:7b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_gpu": 1}
                },
                timeout=120  # Longer timeout for model loading
            )

            if response.ok:
                result = response.json().get("response", "")

                # Parse response
                topic = ""
                keywords = []
                for line in result.split("\n"):
                    if line.startswith("TOPIC:"):
                        topic = line.replace("TOPIC:", "").strip()
                    elif line.startswith("KEYWORDS:"):
                        keywords = [k.strip() for k in line.replace("KEYWORDS:", "").split(",")]

                segment["topic"] = topic or f"Segment {i+1}"
                segment["keywords"] = keywords
                print(f"  [{segment.get('start_sec', 0):>5.0f}s] {segment['topic']}")
            else:
                print(f"  Warning: LLM returned status {response.status_code} for segment {i}")
                segment["topic"] = f"Segment {i+1}"
                segment["keywords"] = []

        except Exception as e:
            print(f"  Warning: LLM failed for segment {i}: {e}")
            segment["topic"] = f"Segment {i+1}"
            segment["keywords"] = []

    return segments


def update_manifest_status(manifest_path: Path, video_id: str, status: str,
                           worker_id: str = None, error: str = None):
    """Update video status in manifest (with file locking)."""
    lock = FileLock(str(manifest_path) + ".lock")

    with lock:
        with open(manifest_path) as f:
            manifest = json.load(f)

        for video in manifest["videos"]:
            if video["id"] == video_id:
                old_status = video["status"]
                video["status"] = status

                if status == "adding_topics":
                    video["topics_worker"] = worker_id
                    video["topics_started_at"] = datetime.now().isoformat()
                elif status == "complete":
                    video["completed_at"] = datetime.now().isoformat()
                    if old_status == "transcribed":
                        manifest["stats"]["transcribed"] = manifest["stats"].get("transcribed", 1) - 1
                    manifest["stats"]["complete"] = manifest["stats"].get("complete", 0) + 1
                elif status == "failed":
                    video["completed_at"] = datetime.now().isoformat()
                    video["error"] = error
                    manifest["stats"]["failed"] = manifest["stats"].get("failed", 0) + 1

                break

        # Check if all done
        if manifest["stats"].get("pending", 0) == 0 and \
           manifest["stats"].get("transcribing", 0) == 0 and \
           manifest["stats"].get("transcribed", 0) == 0:
            if manifest["stats"].get("complete", 0) == manifest["stats"]["total_videos"]:
                manifest["status"] = "complete"
                manifest["completed_at"] = datetime.now().isoformat()

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)


def add_topics_to_video(batch_id: str, video_id: str):
    """
    Add LLM-generated topics to a transcribed video.
    """
    # Get worker info from environment
    ollama_url = os.environ.get("WORKER_OLLAMA_URL", "http://localhost:11434")
    worker_id = os.environ.get("WORKER_NAME", "unknown")

    # Find batch directory
    plan_dir = Path(__file__).parent.parent
    batch_dir = plan_dir / "history" / batch_id
    manifest_path = batch_dir / "manifest.json"
    transcripts_dir = batch_dir / "transcripts"

    if not manifest_path.exists():
        print(f"Error: Batch not found: {batch_id}")
        sys.exit(1)

    # Initialize logger
    logger = BatchLogger(batch_dir / "logs", worker_id=worker_id)

    # Load transcript
    transcript_path = transcripts_dir / f"{video_id}.json"
    if not transcript_path.exists():
        logger.log_worker("TOPICS_ERROR", f"Transcript not found: {video_id}")
        sys.exit(1)

    with open(transcript_path) as f:
        transcript = json.load(f)

    # Check if topics already added
    if transcript.get("topics_added", False):
        logger.log_worker("TOPICS_SKIP", f"Topics already added: {video_id}")
        return

    logger.log_worker("TOPICS_START", f"Adding topics: {transcript['video_title'][:60]}", {
        "video_id": video_id,
        "segments": len(transcript.get("segments", [])),
        "ollama_url": ollama_url
    })

    # Mark as processing
    update_manifest_status(manifest_path, video_id, "adding_topics", worker_id)

    start_time = time.time()

    try:
        # Add topics using LLM
        segments = transcript.get("segments", [])
        segments = identify_topics_with_llm(segments, ollama_url)

        elapsed = time.time() - start_time

        # Update transcript
        transcript["segments"] = segments
        transcript["topics_added"] = True
        transcript["topics_time_sec"] = round(elapsed, 1)
        transcript["topics_worker"] = worker_id
        transcript["completed_at"] = datetime.now().isoformat()

        # Save updated transcript
        with open(transcript_path, 'w') as f:
            json.dump(transcript, f, indent=2)

        # Mark as complete
        update_manifest_status(manifest_path, video_id, "complete")

        # Log completion
        logger.log_worker("TOPICS_COMPLETE",
            f"Added topics to {video_id}: {len(segments)} segments in {elapsed:.1f}s",
            {"video_id": video_id, "segments": len(segments), "time": elapsed})

    except Exception as e:
        elapsed = time.time() - start_time
        logger.log_worker("TOPICS_ERROR", f"Failed: {video_id} - {e}")
        update_manifest_status(manifest_path, video_id, "failed", error=str(e))
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Add LLM topics to a transcribed video")
    parser.add_argument("--batch-id", required=True, help="Batch ID")
    parser.add_argument("--video-id", required=True, help="Video ID")

    args = parser.parse_args()
    add_topics_to_video(args.batch_id, args.video_id)


if __name__ == "__main__":
    main()
