#!/usr/bin/env python3
"""
Video Transcription - Transcribe a SINGLE video from a batch.

This script transcribes ONE video. The brain creates N tasks (one per video)
and workers claim individual tasks. This keeps scripts simple and gives the
brain full visibility into work distribution.

Usage:
    python video_transcribe.py --batch-id abc123 --video-id my_video_id

GPU is auto-detected from CUDA_VISIBLE_DEVICES (set by worker).
"""

import argparse
import os
import sys

# Parse --gpu early to set CUDA_VISIBLE_DEVICES BEFORE any CUDA imports
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--gpu", type=int)
args, _ = parser.parse_known_args()
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from filelock import FileLock

# Add lib folder to path
LIB_PATH = Path(__file__).parent.parent / "lib"
sys.path.insert(0, str(LIB_PATH))

from zim_reader import VideoZIMReader
from transcriber import transcribe_with_timestamps, group_into_chunks, identify_topics_with_llm
from batch_logger import BatchLogger


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

                if status == "processing":
                    video["worker"] = worker_id
                    video["started_at"] = datetime.now().isoformat()
                    manifest["stats"]["pending"] -= 1
                    manifest["stats"]["processing"] += 1
                elif status == "complete":
                    video["completed_at"] = datetime.now().isoformat()
                    manifest["stats"]["processing"] -= 1
                    manifest["stats"]["complete"] += 1
                elif status == "failed":
                    video["completed_at"] = datetime.now().isoformat()
                    video["error"] = error
                    manifest["stats"]["processing"] -= 1
                    manifest["stats"]["failed"] += 1

                break

        # Check if all done
        if manifest["stats"]["pending"] == 0 and manifest["stats"]["processing"] == 0:
            manifest["status"] = "transcription_complete"
            manifest["completed_at"] = datetime.now().isoformat()

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)


def transcribe_single_video(batch_id: str, video_id: str, gpu_id: int):
    """
    Transcribe a single video from the batch manifest.
    """
    worker_id = f"worker-gpu{gpu_id}"

    # Find batch directory
    plan_dir = Path(__file__).parent.parent
    batch_dir = plan_dir / "history" / batch_id
    manifest_path = batch_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"Error: Batch not found: {batch_id}")
        sys.exit(1)

    # Initialize logger
    logger = BatchLogger(batch_dir / "logs", worker_id=worker_id)

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Find the video
    video = None
    for v in manifest["videos"]:
        if v["id"] == video_id:
            video = v
            break

    if not video:
        logger.log_worker("WORKER_ERROR", f"Video not found: {video_id}")
        sys.exit(1)

    # Check if already done
    if video["status"] == "complete":
        logger.log_worker("WORKER_SKIP", f"Already transcribed: {video_id}")
        return

    logger.log_worker("WORKER_START", f"Starting: {video['title'][:60]}", {
        "video_id": video_id,
        "duration": video.get("duration", 0),
        "gpu": gpu_id
    })

    # Mark as processing
    update_manifest_status(manifest_path, video_id, "processing", worker_id)

    start_time = time.time()

    try:
        # Open ZIM and extract video
        reader = VideoZIMReader(manifest["zim_path"])

        try:
            video_content = reader.get_video_content(video["video_path"])
            if not video_content:
                raise Exception(f"Could not extract video from ZIM: {video['video_path']}")

            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
                tmp.write(video_content)
                tmp_path = tmp.name

            try:
                # Transcribe
                raw_segments, duration = transcribe_with_timestamps(
                    tmp_path, gpu_id, manifest["config"]["whisper_model"]
                )

                # Group into chunks
                chunk_duration = manifest["config"]["chunk_duration"]
                chunks = group_into_chunks(raw_segments, chunk_duration)

                # Optionally identify topics
                if not manifest["config"]["skip_llm_topics"]:
                    # Use worker's Ollama URL from environment (set by worker.py)
                    ollama_url = os.environ.get("WORKER_OLLAMA_URL", "http://localhost:11434")
                    chunks = identify_topics_with_llm(chunks, ollama_url=ollama_url)
                else:
                    for i, chunk in enumerate(chunks):
                        chunk["topic"] = f"{video['title']} - Segment {i+1}"
                        chunk["keywords"] = []

                elapsed = time.time() - start_time

                # Build transcript result
                transcript = {
                    "video_id": video_id,
                    "video_title": video["title"],
                    "duration_sec": duration,
                    "segments": [],
                    "transcribe_time_sec": round(elapsed, 1),
                    "worker": worker_id,
                    "gpu": gpu_id,
                    "completed_at": datetime.now().isoformat()
                }

                for i, chunk in enumerate(chunks):
                    transcript["segments"].append({
                        "index": i,
                        "start_sec": chunk["start"],
                        "end_sec": chunk["end"],
                        "text": chunk["text"],
                        "topic": chunk.get("topic", f"Segment {i+1}"),
                        "keywords": chunk.get("keywords", [])
                    })

                # Save transcript
                transcripts_dir = batch_dir / "transcripts"
                transcripts_dir.mkdir(exist_ok=True)
                transcript_path = transcripts_dir / f"{video_id}.json"
                with open(transcript_path, 'w') as f:
                    json.dump(transcript, f, indent=2)

                # Mark complete
                update_manifest_status(manifest_path, video_id, "complete")

                # Log completion with stats
                logger.log_video_complete(
                    video_id=video_id,
                    segments=len(transcript['segments']),
                    duration=duration,
                    transcribe_time=elapsed,
                    worker=worker_id
                )

            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        finally:
            reader.close()

    except Exception as e:
        elapsed = time.time() - start_time
        logger.log_video_error(video_id=video_id, error=str(e), worker=worker_id)
        update_manifest_status(manifest_path, video_id, "failed", error=str(e))
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Transcribe a single video from batch")
    parser.add_argument("--batch-id", required=True, help="Batch ID")
    parser.add_argument("--video-id", required=True, help="Video ID to transcribe")
    parser.add_argument("--gpu", type=int, help="GPU ID (auto-detects from CUDA_VISIBLE_DEVICES)")

    args = parser.parse_args()

    # Determine GPU
    gpu_id = args.gpu
    if gpu_id is None:
        cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_env:
            gpu_id = int(cuda_env.split(",")[0])
        else:
            print("Error: No GPU specified. Use --gpu or set CUDA_VISIBLE_DEVICES")
            sys.exit(1)

    transcribe_single_video(args.batch_id, args.video_id, gpu_id)


if __name__ == "__main__":
    main()
