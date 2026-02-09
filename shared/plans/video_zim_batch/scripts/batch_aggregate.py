#!/usr/bin/env python3
"""
Batch Aggregator - Combine transcripts and generate final output files.

This is Tasks 4-6 in the video_zim_batch plan.
Runs after all workers complete to produce disaster-clippy compatible output.

Usage:
    python batch_aggregate.py --batch-id abc123
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# Add lib folder to path
LIB_PATH = Path(__file__).parent.parent / "lib"
sys.path.insert(0, str(LIB_PATH))

from output import chunks_to_documents, save_metadata, save_index
from embeddings import generate_768_vectors


class BatchLogger:
    """Structured logging for batch operations."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.jsonl_file = self.log_dir / "batch.jsonl"
        self.event_file = self.log_dir / "events.log"

    def log(self, event_type: str, message: str, details: dict = None):
        """Log structured event."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "message": message,
            "details": details or {}
        }

        with open(self.jsonl_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

        with open(self.event_file, 'a') as f:
            f.write(f"[{entry['timestamp']}] [{event_type}] {message}\n")

        print(f"[{event_type}] {message}")


def load_transcripts(batch_dir: Path) -> list:
    """Load all transcript JSONs from the transcripts folder."""
    transcripts = []
    transcript_dir = batch_dir / "transcripts"

    for f in transcript_dir.glob("*.json"):
        with open(f) as fp:
            transcripts.append(json.load(fp))

    return transcripts


def transcripts_to_documents(transcripts: list, source_id: str) -> list:
    """
    Convert all transcripts to disaster-clippy document format.
    """
    all_documents = []

    for transcript in transcripts:
        video_id = transcript["video_id"]
        video_title = transcript["video_title"]

        # Convert transcript segments to chunk format
        chunks = []
        for seg in transcript["segments"]:
            chunks.append({
                "topic": seg.get("topic", f"Segment {seg['index']+1}"),
                "keywords": seg.get("keywords", []),
                "start": seg["start_sec"],
                "end": seg["end_sec"],
                "text": seg["text"],
            })

        # Use local lib function
        docs = chunks_to_documents(chunks, video_id, video_title, source_id)
        all_documents.extend(docs)

    return all_documents


def aggregate_batch(batch_id: str):
    """
    Aggregate all transcripts into final output files.
    """
    # Find batch directory
    plan_dir = Path(__file__).parent.parent
    batch_dir = plan_dir / "history" / batch_id
    manifest_path = batch_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"Error: Batch not found: {batch_id}")
        sys.exit(1)

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Initialize logger
    logger = BatchLogger(batch_dir / "logs")
    logger.log("AGGREGATE_START", f"Starting aggregation for batch {batch_id}")

    # Check if transcription is complete
    stats = manifest["stats"]
    if stats["pending"] > 0 or stats["processing"] > 0:
        logger.log("ERROR", "Transcription not complete", {
            "pending": stats["pending"],
            "processing": stats["processing"]
        })
        print(f"\nError: Transcription not complete!")
        print(f"  Pending: {stats['pending']}")
        print(f"  Processing: {stats['processing']}")
        print(f"  Complete: {stats['complete']}")
        print(f"  Failed: {stats['failed']}")
        sys.exit(1)

    source_id = manifest["source_id"]
    output_folder = Path(manifest["output_folder"])

    # Load all transcripts
    logger.log("LOAD_TRANSCRIPTS", "Loading transcript files...")
    transcripts = load_transcripts(batch_dir)
    logger.log("LOAD_TRANSCRIPTS", f"Loaded {len(transcripts)} transcripts", {
        "count": len(transcripts)
    })

    # Convert to documents
    logger.log("CONVERT_DOCS", "Converting to document format...")
    documents = transcripts_to_documents(transcripts, source_id)
    logger.log("CONVERT_DOCS", f"Created {len(documents)} documents", {
        "document_count": len(documents)
    })

    # Save to batch output folder first
    batch_output = batch_dir / "output"
    batch_output.mkdir(parents=True, exist_ok=True)

    # Save metadata
    logger.log("SAVE_METADATA", "Saving _metadata.json...")
    save_metadata(batch_output, source_id, documents)

    # Save index
    logger.log("SAVE_INDEX", "Saving _index.json...")
    save_index(batch_output, source_id, documents)

    # Generate embeddings (CPU mode for GTX 1060)
    logger.log("GENERATE_EMBEDDINGS", "Generating 768-dim embeddings (CPU mode)...")
    start_time = time.time()

    def progress_callback(current, total, message=""):
        if current % 10 == 0 or current == total:
            print(f"  Embedding progress: {current}/{total} {message}")

    generate_768_vectors(batch_output, source_id, documents, progress_callback)

    embed_time = time.time() - start_time
    logger.log("GENERATE_EMBEDDINGS", f"Embeddings complete in {embed_time:.1f}s", {
        "time_sec": round(embed_time, 1),
        "document_count": len(documents)
    })

    # Verify all files exist
    required_files = ["_metadata.json", "_index.json", "_vectors_768.json"]
    missing = []
    for f in required_files:
        if not (batch_output / f).exists():
            missing.append(f)

    if missing:
        logger.log("ERROR", f"Missing output files: {missing}")
        sys.exit(1)

    # Copy to final output folder
    logger.log("FINALIZE", f"Copying to output folder: {output_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)

    for f in required_files:
        src = batch_output / f
        dst = output_folder / f
        shutil.copy2(src, dst)
        logger.log("COPY", f"Copied {f}", {"size_bytes": src.stat().st_size})

    # Verify document counts match
    with open(output_folder / "_metadata.json") as f:
        meta = json.load(f)
    with open(output_folder / "_index.json") as f:
        idx = json.load(f)
    with open(output_folder / "_vectors_768.json") as f:
        vecs = json.load(f)

    meta_count = meta.get("document_count", len(meta.get("documents", {})))
    idx_count = idx.get("document_count", len(idx.get("documents", {})))
    vec_count = vecs.get("document_count", len(vecs.get("vectors", {})))

    if not (meta_count == idx_count == vec_count):
        logger.log("WARNING", "Document count mismatch!", {
            "metadata": meta_count,
            "index": idx_count,
            "vectors": vec_count
        })
    else:
        logger.log("VERIFY", f"All files have {meta_count} documents")

    # Update manifest as complete
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now().isoformat()
    manifest["result"] = {
        "document_count": len(documents),
        "output_folder": str(output_folder),
        "files": required_files
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.log("COMPLETE", f"Batch {batch_id} complete!", {
        "document_count": len(documents),
        "output_folder": str(output_folder)
    })

    # Print summary
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {batch_id}")
    print(f"{'='*60}")
    print(f"  Videos processed: {stats['complete']}")
    print(f"  Videos failed: {stats['failed']}")
    print(f"  Documents created: {len(documents)}")
    print(f"  Output folder: {output_folder}")
    print(f"\nFiles created:")
    for f in required_files:
        size = (output_folder / f).stat().st_size
        print(f"  {f}: {size:,} bytes")

    print(f"\nThis folder is ready for disaster-clippy!")
    print(f"Copy or symlink to your BACKUP_PATH/{source_id}/ folder.")
    print(f"\nNote: You still need to load vectors into ChromaDB on the")
    print(f"disaster-clippy side. See video-zim-integration.md for code.")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Aggregate batch results")
    parser.add_argument("--batch-id", required=True, help="Batch ID to aggregate")

    args = parser.parse_args()

    aggregate_batch(args.batch_id)


if __name__ == "__main__":
    main()
