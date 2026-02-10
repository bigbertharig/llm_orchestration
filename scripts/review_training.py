#!/usr/bin/env python3
"""
Review and rate training samples for future fine-tuning.

Usage:
  python review_training.py              # Interactive review mode
  python review_training.py --stats      # Show statistics
  python review_training.py --export     # Export rated samples
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# ANSI colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def load_samples(log_path: Path) -> list:
    """Load all training samples."""
    samples = []
    training_log = log_path / "training_samples.jsonl"

    if not training_log.exists():
        return samples

    with open(training_log) as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    return samples


def save_samples(log_path: Path, samples: list):
    """Save all training samples back to file."""
    training_log = log_path / "training_samples.jsonl"

    with open(training_log, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')


def show_stats(samples: list):
    """Show statistics about training samples."""
    total = len(samples)
    rated = sum(1 for s in samples if s.get("human_rating") is not None)
    unrated = total - rated

    outcomes = {}
    for s in samples:
        outcome = s.get("outcome", "unknown")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    sample_types = {}
    for s in samples:
        stype = s.get("sample_type", "unknown")
        sample_types[stype] = sample_types.get(stype, 0) + 1

    ratings = {}
    for s in samples:
        r = s.get("human_rating")
        if r is not None:
            ratings[r] = ratings.get(r, 0) + 1

    print(f"{BOLD}{CYAN}Training Sample Statistics{RESET}")
    print("=" * 50)
    print(f"Total samples: {total}")
    print(f"Rated: {GREEN}{rated}{RESET}")
    print(f"Unrated: {YELLOW}{unrated}{RESET}")

    print(f"\n{BOLD}By Outcome:{RESET}")
    for outcome, count in sorted(outcomes.items()):
        color = GREEN if outcome == "success" else RED if outcome == "failure" else YELLOW
        print(f"  {color}{outcome}: {count}{RESET}")

    print(f"\n{BOLD}By Sample Type:{RESET}")
    for stype, count in sorted(sample_types.items()):
        print(f"  {stype}: {count}")

    if ratings:
        print(f"\n{BOLD}Human Ratings:{RESET}")
        for rating, count in sorted(ratings.items()):
            stars = "★" * rating + "☆" * (5 - rating)
            print(f"  {stars} ({rating}): {count}")


def review_sample(sample: dict, idx: int, total: int) -> dict:
    """Review a single sample and get human feedback."""
    clear_screen()

    print(f"{BOLD}{CYAN}Sample {idx + 1} of {total}{RESET}")
    print("=" * 70)

    print(f"{BOLD}Type:{RESET} {sample.get('sample_type', '?')}")
    print(f"{BOLD}Model:{RESET} {sample.get('model', '?')}")
    print(f"{BOLD}Outcome:{RESET} {sample.get('outcome', '?')}")
    print(f"{BOLD}Time:{RESET} {sample.get('timestamp', '?')}")

    print(f"\n{BOLD}Prompt:{RESET}")
    print(f"{DIM}{'-' * 70}{RESET}")
    prompt = sample.get("prompt", "(no prompt)")
    print(prompt[:500] + ("..." if len(prompt) > 500 else ""))

    if sample.get("context"):
        print(f"\n{BOLD}Context:{RESET}")
        context = sample.get("context", "")
        print(context[:200] + ("..." if len(context) > 200 else ""))

    print(f"\n{BOLD}Response:{RESET}")
    print(f"{DIM}{'-' * 70}{RESET}")
    response = sample.get("response", "(no response)")
    print(response[:1000] + ("..." if len(response) > 1000 else ""))

    print(f"\n{DIM}{'-' * 70}{RESET}")

    # Get rating
    current_rating = sample.get("human_rating")
    rating_str = f" (current: {current_rating})" if current_rating else ""

    while True:
        try:
            inp = input(f"\n{BOLD}Rate 1-5 (or s=skip, q=quit){rating_str}: {RESET}").strip().lower()

            if inp == 'q':
                return None  # Signal to quit
            elif inp == 's' or inp == '':
                return sample  # Skip, no change
            elif inp in ['1', '2', '3', '4', '5']:
                sample["human_rating"] = int(inp)
                break
            else:
                print(f"{RED}Invalid input{RESET}")
        except EOFError:
            return None

    # Get optional feedback
    feedback = input(f"{BOLD}Feedback (optional, enter to skip): {RESET}").strip()
    if feedback:
        sample["human_feedback"] = feedback

    # For low ratings, optionally get preferred response
    if sample["human_rating"] <= 2:
        preferred = input(f"{BOLD}Preferred response? (enter to skip): {RESET}").strip()
        if preferred:
            sample["preferred_response"] = preferred

    return sample


def interactive_review(log_path: Path):
    """Interactive review mode."""
    samples = load_samples(log_path)

    if not samples:
        print(f"{YELLOW}No training samples found.{RESET}")
        return

    # Filter to unrated samples
    unrated = [s for s in samples if s.get("human_rating") is None]

    print(f"Found {len(samples)} total samples, {len(unrated)} unrated.")
    inp = input("Review (u)nrated only or (a)ll? [u]: ").strip().lower()

    to_review = unrated if inp != 'a' else samples

    if not to_review:
        print(f"{GREEN}All samples have been rated!{RESET}")
        return

    print(f"\nReviewing {len(to_review)} samples. Press 'q' to quit and save.\n")
    input("Press Enter to start...")

    reviewed = 0
    for i, sample in enumerate(to_review):
        result = review_sample(sample, i, len(to_review))

        if result is None:  # Quit
            break

        # Update in main samples list
        for j, s in enumerate(samples):
            if s.get("id") == sample.get("id"):
                samples[j] = result
                break

        reviewed += 1

    # Save
    save_samples(log_path, samples)
    print(f"\n{GREEN}Saved {reviewed} reviewed samples.{RESET}")


def export_rated(log_path: Path, output_path: Path = None):
    """Export rated samples in training format."""
    samples = load_samples(log_path)

    rated = [s for s in samples if s.get("human_rating") is not None]

    if not rated:
        print(f"{YELLOW}No rated samples to export.{RESET}")
        return

    # Group by quality
    good = [s for s in rated if s["human_rating"] >= 4]
    acceptable = [s for s in rated if s["human_rating"] == 3]
    poor = [s for s in rated if s["human_rating"] <= 2]

    output_path = output_path or log_path / "training_export"
    output_path.mkdir(parents=True, exist_ok=True)

    # Export good samples as positive examples
    with open(output_path / "positive_examples.jsonl", 'w') as f:
        for s in good:
            example = {
                "prompt": s.get("prompt", ""),
                "context": s.get("context", ""),
                "response": s.get("response", ""),
                "rating": s.get("human_rating"),
            }
            f.write(json.dumps(example) + "\n")

    # Export poor samples with corrections as preference pairs
    with open(output_path / "preference_pairs.jsonl", 'w') as f:
        for s in poor:
            if s.get("preferred_response"):
                pair = {
                    "prompt": s.get("prompt", ""),
                    "context": s.get("context", ""),
                    "rejected": s.get("response", ""),
                    "chosen": s.get("preferred_response", ""),
                }
                f.write(json.dumps(pair) + "\n")

    print(f"{GREEN}Exported:{RESET}")
    print(f"  {len(good)} positive examples -> {output_path / 'positive_examples.jsonl'}")
    print(f"  {len([s for s in poor if s.get('preferred_response')])} preference pairs -> {output_path / 'preference_pairs.jsonl'}")


def main():
    parser = argparse.ArgumentParser(description="Review training samples")
    default_config = str(Path(__file__).resolve().parent.parent / "shared" / "agents" / "config.json")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--export", action="store_true", help="Export rated samples")
    args = parser.parse_args()

    config = load_config(args.config)
    log_path = Path(config["shared_path"]) / "logs"

    if args.stats:
        samples = load_samples(log_path)
        show_stats(samples)
    elif args.export:
        export_rated(log_path)
    else:
        interactive_review(log_path)


if __name__ == "__main__":
    main()
