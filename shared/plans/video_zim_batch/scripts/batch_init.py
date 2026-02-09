#!/usr/bin/env python3
"""
Batch Initializer - Read ZIM file and create batch manifest.

This is Task 1 in the video_zim_batch plan.
Creates the manifest.json that workers will use to claim videos.

Usage:
    python batch_init.py --zim /path/to/file.zim --source-id my-source --output /path/to/output
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add lib folder to path
LIB_PATH = Path(__file__).parent.parent / "lib"
sys.path.insert(0, str(LIB_PATH))

from zim_reader import VideoZIMReader
from batch_logger import BatchLogger


def init_batch(zim_path: str, source_id: str, output_folder: str,
               chunk_duration: float = 60.0, skip_llm: bool = False,
               batch_id: str = None) -> dict:
    """
    Initialize a batch processing run.

    Creates manifest with video list and batch configuration.
    """
    if batch_id is None:
        batch_id = str(uuid.uuid4())[:8]

    # Create batch directory structure
    plan_dir = Path(__file__).parent.parent
    batch_dir = plan_dir / "history" / batch_id

    (batch_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    (batch_dir / "output").mkdir(parents=True, exist_ok=True)
    (batch_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = BatchLogger(batch_dir / "logs")
    logger.log_batch("BATCH_INIT", f"Initializing batch {batch_id}", {
        "zim_path": zim_path,
        "source_id": source_id,
        "output_folder": output_folder
    })

    # Read ZIM file to list videos
    logger.log_batch("ZIM_READ", f"Opening ZIM file: {zim_path}")

    try:
        reader = VideoZIMReader(zim_path)
        videos = reader.list_videos()
        reader.close()

        logger.log_batch("ZIM_READ", f"Found {len(videos)} videos in ZIM", {
            "video_count": len(videos)
        })

    except Exception as e:
        logger.log_batch("BATCH_ERROR", f"Failed to read ZIM: {e}", {"error": str(e)})
        raise

    # Build video list for manifest
    video_list = []
    total_duration = 0

    for v in videos:
        video_list.append({
            "id": v["id"],
            "title": v["title"],
            "duration": v.get("duration", 0),
            "video_path": v.get("video_path", ""),
            "status": "pending",
            "worker": None,
            "started_at": None,
            "completed_at": None
        })
        total_duration += v.get("duration", 0)

    # Create manifest
    manifest = {
        "batch_id": batch_id,
        "zim_path": str(zim_path),
        "source_id": source_id,
        "output_folder": str(output_folder),
        "config": {
            "chunk_duration": chunk_duration,
            "whisper_model": "small.en",
            "skip_llm_topics": skip_llm,
            "embedding_model": "all-mpnet-base-v2",
            "embedding_device": "cpu"  # GTX 1060 PyTorch limitation
        },
        "videos": video_list,
        "stats": {
            "total_videos": len(video_list),
            "total_duration_sec": total_duration,
            "pending": len(video_list),
            "processing": 0,
            "complete": 0,
            "failed": 0
        },
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None
    }

    # Save manifest
    manifest_path = batch_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.log_batch("BATCH_INIT_COMPLETE", f"Batch initialized: {batch_id}", {
        "manifest_path": str(manifest_path),
        "video_count": len(video_list),
        "total_duration_min": round(total_duration / 60, 1)
    })

    # Print summary
    print(f"\n{'='*60}")
    print(f"BATCH INITIALIZED: {batch_id}")
    print(f"{'='*60}")
    print(f"  ZIM: {zim_path}")
    print(f"  Source ID: {source_id}")
    print(f"  Videos: {len(video_list)}")
    print(f"  Total duration: {total_duration // 60}m {total_duration % 60}s")
    print(f"  Output: {output_folder}")
    print(f"\nBatch directory: {batch_dir}")
    print(f"\nNext steps:")
    print(f"  1. Start workers:")
    print(f"     python video_transcribe.py --batch-id {batch_id} --gpu 1")
    print(f"     python video_transcribe.py --batch-id {batch_id} --gpu 2")
    print(f"     python video_transcribe.py --batch-id {batch_id} --gpu 4")
    print(f"  2. After all videos complete:")
    print(f"     python batch_aggregate.py --batch-id {batch_id}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Initialize video batch processing")
    parser.add_argument("--zim", required=True, help="Path to ZIM file")
    parser.add_argument("--source-id", required=True, help="Source identifier")
    parser.add_argument("--output", required=True, help="Output folder for final files")
    parser.add_argument("--batch-id", help="Batch ID (auto-generated if not provided)")
    parser.add_argument("--chunk-duration", type=float, default=60.0,
                        help="Segment duration in seconds (default: 60)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM topic identification")

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.zim).exists():
        print(f"Error: ZIM file not found: {args.zim}")
        sys.exit(1)

    # Create output folder
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Run initialization
    manifest = init_batch(
        zim_path=args.zim,
        source_id=args.source_id,
        output_folder=args.output,
        chunk_duration=args.chunk_duration,
        skip_llm=args.skip_llm,
        batch_id=args.batch_id
    )

    return manifest


if __name__ == "__main__":
    main()
