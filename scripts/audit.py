#!/usr/bin/env python3
"""
Audit tool for Claude Code to review brain decisions and worker outputs.
Run after tests to verify the brain's judgments were correct.

Usage:
  python audit.py                    # Full audit report
  python audit.py --decisions        # Review brain decisions only
  python audit.py --evaluations      # Review brain's worker evaluations
  python audit.py --disagree         # Find cases to disagree with brain
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def load_decisions(log_path: Path) -> list:
    """Load brain decision log."""
    decisions = []
    decision_log = log_path / "brain_decisions.log"

    if decision_log.exists():
        with open(decision_log) as f:
            for line in f:
                if line.strip():
                    try:
                        decisions.append(json.loads(line))
                    except Exception:
                        pass  # Skip malformed log line
    return decisions


def load_training_samples(log_path: Path) -> list:
    """Load training samples."""
    samples = []
    training_log = log_path / "training_samples.jsonl"

    if training_log.exists():
        with open(training_log) as f:
            for line in f:
                if line.strip():
                    try:
                        samples.append(json.loads(line))
                    except Exception:
                        pass  # Skip malformed log line
    return samples


def load_completed_tasks(shared_path: Path) -> list:
    """Load completed tasks."""
    tasks = []
    for task_file in (shared_path / "tasks" / "complete").glob("*.json"):
        try:
            with open(task_file) as f:
                tasks.append(json.load(f))
        except Exception:
            pass  # Skip malformed task file
    return tasks


def print_section(title: str):
    print(f"\n{CYAN}{BOLD}{'=' * 70}{RESET}")
    print(f"{CYAN}{BOLD}{title}{RESET}")
    print(f"{CYAN}{BOLD}{'=' * 70}{RESET}\n")


def audit_decisions(decisions: list):
    """Analyze brain decisions."""
    print_section("BRAIN DECISION AUDIT")

    by_type = defaultdict(list)
    for d in decisions:
        by_type[d.get("type", "unknown")].append(d)

    print(f"Total decisions: {len(decisions)}\n")

    for dtype, items in sorted(by_type.items()):
        print(f"{BOLD}{dtype}{RESET}: {len(items)}")

    # Show decision timeline
    print(f"\n{BOLD}Decision Timeline:{RESET}")
    for d in decisions[-20:]:  # Last 20
        ts = d.get("timestamp", "")[-8:]
        dtype = d.get("type", "?")
        msg = d.get("message", "")[:60]
        print(f"  {ts} [{dtype:20}] {msg}")


def audit_evaluations(samples: list):
    """Analyze brain's evaluations of worker outputs."""
    print_section("BRAIN EVALUATION AUDIT")

    evaluations = [s for s in samples if s.get("sample_type") == "worker_evaluation"]

    if not evaluations:
        print("No evaluations found yet.")
        return

    print(f"Total evaluations: {len(evaluations)}\n")

    # Group by outcome
    good = [e for e in evaluations if e.get("outcome") == "good"]
    bad = [e for e in evaluations if e.get("outcome") == "bad"]

    print(f"{GREEN}Accepted (good):{RESET} {len(good)}")
    print(f"{RED}Rejected (bad):{RESET} {len(bad)}")

    # Show ratings distribution
    ratings = defaultdict(int)
    for e in evaluations:
        r = e.get("metadata", {}).get("brain_rating", 0)
        ratings[r] += 1

    print(f"\n{BOLD}Rating Distribution:{RESET}")
    for r in range(1, 6):
        count = ratings.get(r, 0)
        bar = "█" * count
        print(f"  {r}/5: {bar} ({count})")

    # Show rejected outputs for review
    if bad:
        print(f"\n{BOLD}{RED}Rejected Outputs (review these):{RESET}")
        for e in bad[:5]:
            worker = e.get("metadata", {}).get("worker", "?")
            feedback = e.get("metadata", {}).get("brain_feedback", "")
            prompt = e.get("prompt", "")[:50]
            response = e.get("response", "")[:100]

            print(f"\n  {YELLOW}Worker:{RESET} {worker}")
            print(f"  {YELLOW}Prompt:{RESET} {prompt}...")
            print(f"  {YELLOW}Response:{RESET} {response}...")
            print(f"  {YELLOW}Brain said:{RESET} {feedback}")
            print(f"  {YELLOW}Issues:{RESET} {e.get('metadata', {}).get('issues', [])}")


def find_disagreements(samples: list, completed_tasks: list):
    """Find cases where Claude Code might disagree with brain."""
    print_section("POTENTIAL DISAGREEMENTS")

    evaluations = [s for s in samples if s.get("sample_type") == "worker_evaluation"]

    suspicious = []

    for e in evaluations:
        meta = e.get("metadata", {})
        rating = meta.get("brain_rating", 3)
        issues = meta.get("issues", [])
        response = e.get("response", "")

        # Cases to flag:
        # 1. Low rating but response looks ok (has content, structured)
        if rating <= 2 and len(response) > 100 and not any(err in response.lower() for err in ["error", "failed", "exception"]):
            suspicious.append({"type": "harsh_rejection", "sample": e, "reason": "Response has content but was rejected"})

        # 2. High rating but response is very short
        if rating >= 4 and len(response) < 50:
            suspicious.append({"type": "easy_pass", "sample": e, "reason": "Short response got high rating"})

        # 3. Accepted with issues flagged
        if e.get("outcome") == "good" and len(issues) > 0:
            suspicious.append({"type": "issues_ignored", "sample": e, "reason": f"Accepted despite issues: {issues}"})

    if not suspicious:
        print(f"{GREEN}No obvious disagreements found.{RESET}")
        return

    print(f"Found {len(suspicious)} cases to review:\n")

    for i, s in enumerate(suspicious[:10]):
        e = s["sample"]
        print(f"{BOLD}{i+1}. {s['type'].upper()}{RESET}")
        print(f"   Reason: {s['reason']}")
        print(f"   Worker: {e.get('metadata', {}).get('worker', '?')}")
        print(f"   Brain rating: {e.get('metadata', {}).get('brain_rating', '?')}/5")
        print(f"   Prompt: {e.get('prompt', '')[:60]}...")
        print(f"   Response: {e.get('response', '')[:80]}...")
        print()


def full_audit(config_path: str):
    """Run full audit."""
    config = load_config(config_path)
    shared_path = Path(config["shared_path"])
    log_path = shared_path / "logs"

    decisions = load_decisions(log_path)
    samples = load_training_samples(log_path)
    tasks = load_completed_tasks(shared_path)

    print(f"{BOLD}{CYAN}AGENT SYSTEM AUDIT REPORT{RESET}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {config_path}")

    audit_decisions(decisions)
    audit_evaluations(samples)
    find_disagreements(samples, tasks)

    # Summary
    print_section("SUMMARY FOR CLAUDE CODE REVIEW")
    print("1. Check 'Rejected Outputs' - did brain reject good work?")
    print("2. Check 'Potential Disagreements' - brain might be wrong")
    print("3. Use review_training.py to override brain ratings if needed")
    print("4. Export corrected samples for future fine-tuning")


def main():
    parser = argparse.ArgumentParser(description="Audit agent system")
    default_config = str(Path(__file__).resolve().parent.parent / "shared" / "agents" / "config.json")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--decisions", action="store_true", help="Decisions only")
    parser.add_argument("--evaluations", action="store_true", help="Evaluations only")
    parser.add_argument("--disagree", action="store_true", help="Find disagreements")
    args = parser.parse_args()

    config = load_config(args.config)
    shared_path = Path(config["shared_path"])
    log_path = shared_path / "logs"

    if args.decisions:
        decisions = load_decisions(log_path)
        audit_decisions(decisions)
    elif args.evaluations:
        samples = load_training_samples(log_path)
        audit_evaluations(samples)
    elif args.disagree:
        samples = load_training_samples(log_path)
        tasks = load_completed_tasks(shared_path)
        find_disagreements(samples, tasks)
    else:
        full_audit(args.config)


if __name__ == "__main__":
    main()
