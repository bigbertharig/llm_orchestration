#!/usr/bin/env python3
"""
Video Transcription (Whisper Only) - Transcribe a SINGLE video from a batch.

This script ONLY runs Whisper transcription. Topic identification is done
in a separate LLM task so the brain can manage model loading properly.

Usage:
    python video_transcribe_whisper.py --batch-id abc123 --video-id my_video_id

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
from transcriber import transcribe_with_timestamps, group_into_chunks
from batch_logger import BatchLogger


def recompute_manifest_stats(manifest: dict) -> None:
    """Rebuild stats from current video statuses to prevent counter drift."""
    videos = manifest.get("videos", [])
    stats = manifest.get("stats", {})
    total_duration = sum(v.get("duration", 0) or 0 for v in videos)

    rebuilt = {
        "total_videos": len(videos),
        "total_duration_sec": total_duration,
        "pending": 0,
        "processing": 0,
        "complete": 0,
        "failed": 0,
        "transcribing": 0,
        "transcribed": 0,
        "adding_topics": 0
    }

    for video in videos:
        status = video.get("status", "pending")
        if status in rebuilt:
            rebuilt[status] += 1
        elif status == "processing":
            rebuilt["processing"] += 1
        else:
            # Unknown statuses are treated as pending for safety.
            rebuilt["pending"] += 1

    stats.update(rebuilt)
    manifest["stats"] = stats


def update_manifest_status(manifest_path: Path, video_id: str, status: str,
                           worker_id: str = None, error: str = None):
    """Update video status in manifest (with file locking)."""
    lock = FileLock(str(manifest_path) + ".lock")

    with lock:
        with open(manifest_path) as f:
            manifest = json.load(f)

        for video in manifest["videos"]:
            if video["id"] == video_id:
                old_status = video.get("status")
                if old_status == status:
                    break
                video["status"] = status

                if status == "transcribing":
                    video["worker"] = worker_id
                    video["started_at"] = datetime.now().isoformat()
                elif status == "transcribed":
                    video["transcribed_at"] = datetime.now().isoformat()
                    video["error"] = None
                elif status == "failed":
                    video["completed_at"] = datetime.now().isoformat()
                    video["error"] = error

                break

        recompute_manifest_stats(manifest)

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)


def transcribe_single_video(batch_id: str, video_id: str, gpu_id: int):
    """
    Transcribe a single video from the batch manifest (Whisper only, no LLM).
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
    if video["status"] in ["transcribed", "complete"]:
        logger.log_worker("WORKER_SKIP", f"Already transcribed: {video_id}")
        return

    logger.log_worker("WORKER_START", f"Transcribing: {video['title'][:60]}", {
        "video_id": video_id,
        "duration": video.get("duration", 0),
        "gpu": gpu_id
    })

    # Mark as transcribing
    update_manifest_status(manifest_path, video_id, "transcribing", worker_id)

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
                # Transcribe with Whisper
                raw_segments, duration = transcribe_with_timestamps(
                    tmp_path, gpu_id, manifest["config"]["whisper_model"]
                )

                # Group into chunks (no LLM topics yet - just placeholders)
                chunk_duration = manifest["config"]["chunk_duration"]
                chunks = group_into_chunks(raw_segments, chunk_duration)

                # Add placeholder topics (LLM step will fill these in)
                for i, chunk in enumerate(chunks):
                    chunk["topic"] = None  # Will be filled by LLM task
                    chunk["keywords"] = []

                elapsed = time.time() - start_time

                # Build transcript result (without LLM topics)
                transcript = {
                    "video_id": video_id,
                    "video_title": video["title"],
                    "duration_sec": duration,
                    "segments": [],
                    "transcribe_time_sec": round(elapsed, 1),
                    "worker": worker_id,
                    "gpu": gpu_id,
                    "transcribed_at": datetime.now().isoformat(),
                    "topics_added": False  # Flag for LLM task
                }

                for i, chunk in enumerate(chunks):
                    transcript["segments"].append({
                        "index": i,
                        "start_sec": chunk["start"],
                        "end_sec": chunk["end"],
                        "text": chunk["text"],
                        "topic": None,  # Will be filled by LLM task
                        "keywords": []
                    })

                # Save transcript
                transcripts_dir = batch_dir / "transcripts"
                transcripts_dir.mkdir(exist_ok=True)
                transcript_path = transcripts_dir / f"{video_id}.json"
                with open(transcript_path, 'w') as f:
                    json.dump(transcript, f, indent=2)

                # Mark as transcribed (ready for LLM step)
                update_manifest_status(manifest_path, video_id, "transcribed")

                # Log completion with stats
                logger.log_worker("WHISPER_COMPLETE",
                    f"Transcribed {video_id}: {len(transcript['segments'])} segments in {elapsed:.1f}s",
                    {"video_id": video_id, "segments": len(transcript['segments']),
                     "duration": duration, "time": elapsed})

            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        finally:
            reader.close()

    except Exception as e:
        elapsed = time.time() - start_time
        logger.log_worker("WHISPER_ERROR", f"Failed: {video_id} - {e}")
        update_manifest_status(manifest_path, video_id, "failed", error=str(e))
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Transcribe a single video (Whisper only)")
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
