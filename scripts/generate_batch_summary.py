#!/usr/bin/env python3
"""
Generate execution summary for a completed batch.

Analyzes all completed and failed tasks to create:
1. EXECUTION_SUMMARY.md - Human-readable lessons learned
2. execution_stats.json - Machine-readable statistics

This script is automatically inserted by the brain at the end of every plan.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any


def load_tasks_from_folder(folder_path: Path) -> List[Dict[str, Any]]:
    """Load all task JSON files from a folder."""
    tasks = []
    if not folder_path.exists():
        return tasks

    for task_file in folder_path.glob("*.json"):
        try:
            with open(task_file) as f:
                task = json.load(f)
                tasks.append(task)
        except Exception as e:
            print(f"Warning: Failed to load {task_file}: {e}")

    return tasks


def analyze_batch(batch_id: str, plan_name: str, plan_dir: Path, shared_path: Path) -> Dict[str, Any]:
    """Analyze all tasks for this batch."""

    # Load completed and failed tasks
    tasks_path = shared_path / "tasks"
    completed_tasks = load_tasks_from_folder(tasks_path / "complete")
    failed_tasks = load_tasks_from_folder(tasks_path / "failed")

    # Filter tasks for this batch
    batch_completed = [t for t in completed_tasks if t.get("batch_id") == batch_id]
    batch_failed = [t for t in failed_tasks if t.get("batch_id") == batch_id]

    all_tasks = batch_completed + batch_failed

    if not all_tasks:
        return {
            "error": "No tasks found for this batch",
            "batch_id": batch_id,
            "completed": 0,
            "failed": 0
        }

    # Overall stats
    total_tasks = len(all_tasks)
    succeeded = len(batch_completed)
    failed = len(batch_failed)

    # Find start/end times
    start_times = [datetime.fromisoformat(t.get("created_at", datetime.now().isoformat()))
                   for t in all_tasks if t.get("created_at")]
    end_times = [datetime.fromisoformat(t.get("completed_at", datetime.now().isoformat()))
                 for t in all_tasks if t.get("completed_at")]

    started_at = min(start_times) if start_times else datetime.now()
    completed_at = max(end_times) if end_times else datetime.now()
    duration_seconds = (completed_at - started_at).total_seconds()

    # Task class breakdown
    by_class = defaultdict(lambda: {"total": 0, "success": 0, "fail": 0, "durations": [], "vram_est": [], "vram_act": []})

    for task in all_tasks:
        task_class = task.get("task_class", "cpu")
        by_class[task_class]["total"] += 1

        if task in batch_completed:
            by_class[task_class]["success"] += 1

            # Track duration
            if task.get("completed_at") and task.get("started_at"):
                task_start = datetime.fromisoformat(task["started_at"])
                task_end = datetime.fromisoformat(task["completed_at"])
                duration = (task_end - task_start).total_seconds()
                by_class[task_class]["durations"].append(duration)

            # Track VRAM (from task result if available)
            result = task.get("result", {})
            if isinstance(result, dict):
                vram_used = result.get("max_vram_used_mb", 0)
                if vram_used > 0:
                    by_class[task_class]["vram_act"].append(vram_used)
        else:
            by_class[task_class]["fail"] += 1

    # Calculate averages
    class_stats = {}
    for tc, stats in by_class.items():
        avg_duration = sum(stats["durations"]) / len(stats["durations"]) if stats["durations"] else 0
        avg_vram = sum(stats["vram_act"]) / len(stats["vram_act"]) if stats["vram_act"] else 0

        class_stats[tc] = {
            "total": stats["total"],
            "success": stats["success"],
            "fail": stats["fail"],
            "avg_duration_s": round(avg_duration, 1),
            "avg_vram_mb": round(avg_vram, 0) if avg_vram > 0 else None
        }

    # Worker performance
    by_worker = defaultdict(lambda: {"tasks": 0, "failures": 0, "resource_constraints": []})

    for task in all_tasks:
        worker = task.get("assigned_to") or task.get("workers_attempted", ["unknown"])[-1] if task.get("workers_attempted") else "unknown"
        by_worker[worker]["tasks"] += 1

        if task in batch_failed:
            by_worker[worker]["failures"] += 1

    worker_stats = dict(by_worker)

    # Retry analysis
    retried_tasks = [t for t in all_tasks if t.get("attempts", 1) > 1]

    # Failures analysis
    failures = []
    for task in batch_failed:
        result = task.get("result", {})
        error = result.get("error", "Unknown error") if isinstance(result, dict) else "Unknown error"

        failures.append({
            "task_id": task.get("task_id", "unknown")[:8],
            "task_name": task.get("name", "unknown"),
            "error": error[:200],  # Truncate long errors
            "attempts": task.get("attempts", 1),
            "workers_attempted": task.get("workers_attempted", [])
        })

    # Brain interventions (rough estimate from task metadata)
    definition_fixes = len([t for t in all_tasks if t.get("definition_error_fixed")])

    return {
        "batch_id": batch_id,
        "plan_name": plan_name,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": int(duration_seconds),
        "overall": {
            "total_tasks": total_tasks,
            "succeeded": succeeded,
            "failed": failed,
            "retried": len(retried_tasks),
            "success_rate": round(100 * succeeded / total_tasks, 1) if total_tasks > 0 else 0
        },
        "by_class": class_stats,
        "by_worker": worker_stats,
        "brain_interventions": {
            "definition_fixes": definition_fixes
        },
        "failures": failures
    }


def format_duration(seconds: int) -> str:
    """Format duration as human-readable string."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def generate_markdown_summary(analysis: Dict[str, Any]) -> str:
    """Generate human-readable markdown summary."""

    if "error" in analysis:
        return f"# Execution Summary: {analysis['batch_id']}\n\nError: {analysis['error']}\n"

    overall = analysis["overall"]
    by_class = analysis["by_class"]
    by_worker = analysis["by_worker"]
    failures = analysis["failures"]

    md = f"""# Execution Summary: {analysis['plan_name']}

**Batch ID:** {analysis['batch_id']}
**Started:** {analysis['started_at']}
**Completed:** {analysis['completed_at']}
**Duration:** {format_duration(analysis['duration_seconds'])}

## Overall Results

- Total tasks: {overall['total_tasks']}
- Succeeded: {overall['succeeded']} ({overall['success_rate']}%)
- Failed: {overall['failed']}
- Retried: {overall['retried']} tasks

## Task Class Performance

| Class | Total | Success | Failed | Avg Duration | Avg VRAM |
|-------|-------|---------|--------|--------------|----------|
"""

    for tc, stats in sorted(by_class.items()):
        vram_str = f"{stats['avg_vram_mb']}MB" if stats['avg_vram_mb'] else "N/A"
        md += f"| {tc} | {stats['total']} | {stats['success']} | {stats['fail']} | {stats['avg_duration_s']}s | {vram_str} |\n"

    md += "\n## Worker Performance\n\n"
    md += "| Worker | Tasks Completed | Failures |\n"
    md += "|--------|----------------|----------|\n"

    for worker, stats in sorted(by_worker.items()):
        md += f"| {worker} | {stats['tasks']} | {stats['failures']} |\n"

    md += f"\n## Brain Interventions\n\n"
    md += f"- Definition fixes: {analysis['brain_interventions']['definition_fixes']}\n"

    if failures:
        md += f"\n## Failures Analysis\n\n"
        for f in failures[:10]:  # Limit to first 10 failures
            md += f"### Task: {f['task_name']} (ID: {f['task_id']})\n\n"
            md += f"- **Error:** {f['error']}\n"
            md += f"- **Attempts:** {f['attempts']}\n"
            md += f"- **Workers attempted:** {', '.join(f['workers_attempted']) if f['workers_attempted'] else 'None'}\n\n"

    md += "\n## Lessons Learned\n\n"

    # Auto-generate some lessons based on data
    if overall['success_rate'] >= 95:
        md += "- High success rate indicates plan is well-designed and robust\n"

    if overall['failed'] > overall['total_tasks'] * 0.1:
        md += f"- {overall['failed']} failures ({100 - overall['success_rate']:.1f}%) indicates issues that need investigation\n"

    if overall['retried'] > 0:
        md += f"- {overall['retried']} tasks required retries - consider improving error handling\n"

    # Check for worker imbalance
    if len(by_worker) > 1:
        task_counts = [stats['tasks'] for stats in by_worker.values()]
        max_tasks = max(task_counts)
        min_tasks = min(task_counts)
        if max_tasks > min_tasks * 2:
            md += "- Worker load imbalance detected - some workers handled significantly more tasks\n"

    md += "\n---\n\n"
    md += "*Auto-generated by generate_batch_summary.py*\n"

    return md


def main():
    parser = argparse.ArgumentParser(description="Generate batch execution summary")
    parser.add_argument("--batch-id", required=True, help="Batch ID")
    parser.add_argument("--plan-name", required=True, help="Plan name")
    parser.add_argument("--plan-dir", required=True, help="Plan directory path")

    args = parser.parse_args()

    plan_dir = Path(args.plan_dir)

    # Find shared path (go up from plan dir)
    shared_path = plan_dir.parent

    print(f"Generating execution summary for batch {args.batch_id}")
    print(f"Plan: {args.plan_name}")
    print(f"Plan dir: {plan_dir}")
    print(f"Shared path: {shared_path}")

    # Analyze the batch
    analysis = analyze_batch(args.batch_id, args.plan_name, plan_dir, shared_path)

    # Output directory
    output_dir = plan_dir / "history" / args.batch_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate markdown summary
    markdown = generate_markdown_summary(analysis)
    summary_file = output_dir / "EXECUTION_SUMMARY.md"
    with open(summary_file, 'w') as f:
        f.write(markdown)
    print(f"Wrote {summary_file}")

    # Generate JSON stats
    stats_file = output_dir / "execution_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Wrote {stats_file}")

    # Print summary to stdout
    print("\n" + "="*60)
    print(markdown)
    print("="*60)

    return 0


if __name__ == "__main__":
    exit(main())
