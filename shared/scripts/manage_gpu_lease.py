#!/usr/bin/env python3
"""Acquire, inspect, renew, and release controller-neutral GPU leases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _agents_path(shared_path: Path) -> Path:
    return shared_path / "agents"


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _emit(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage shared GPU leases")
    parser.add_argument("--shared-path", default="/mnt/shared")
    subparsers = parser.add_subparsers(dest="command", required=True)

    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--controller", required=True)
    acquire.add_argument("--owner", required=True)
    acquire.add_argument("--run-id", required=True)
    acquire.add_argument("--gpus", required=True, help="Comma-separated physical GPU IDs")
    acquire.add_argument("--ports", default="", help="Optional comma-separated runtime ports")
    acquire.add_argument("--lane", default="")
    acquire.add_argument("--ttl-seconds", type=int, default=120)

    renew = subparsers.add_parser("renew")
    renew.add_argument("--lease-id", required=True)
    renew.add_argument("--owner", required=True)
    renew.add_argument("--ttl-seconds", type=int, default=120)

    release = subparsers.add_parser("release")
    release.add_argument("--lease-id", required=True)
    release.add_argument("--owner", default="")
    release.add_argument(
        "--force-reconciled",
        action="store_true",
        help="Release without owner match only after process/port/runtime reconciliation",
    )

    show = subparsers.add_parser("show")
    show.add_argument("--lease-id")
    show.add_argument("--gpu", type=int)

    subparsers.add_parser("list")

    args = parser.parse_args()
    shared_path = Path(args.shared_path).resolve()
    sys.path.insert(0, str(_agents_path(shared_path)))

    from gpu_leases import GPULeaseError, GPULeaseStore

    store = GPULeaseStore(shared_path)
    try:
        if args.command == "acquire":
            lease = store.acquire(
                controller=args.controller,
                owner=args.owner,
                gpu_ids=_csv_ints(args.gpus),
                run_id=args.run_id,
                lane=args.lane or None,
                ttl_seconds=args.ttl_seconds,
                ports=_csv_ints(args.ports),
            )
            _emit(lease)
            return 0
        if args.command == "renew":
            _emit(
                store.renew(
                    args.lease_id,
                    owner=args.owner,
                    ttl_seconds=args.ttl_seconds,
                )
            )
            return 0
        if args.command == "release":
            if args.force_reconciled and args.owner:
                parser.error("--force-reconciled and --owner are mutually exclusive")
            store.release(
                args.lease_id,
                owner=args.owner,
                force=args.force_reconciled,
            )
            _emit({"released": args.lease_id})
            return 0
        if args.command == "show":
            if bool(args.lease_id) == (args.gpu is not None):
                parser.error("show requires exactly one of --lease-id or --gpu")
            lease = store.get(args.lease_id) if args.lease_id else store.for_gpu(args.gpu)
            _emit({"lease": lease})
            return 0 if lease else 1

        leases = store.list_leases()
        _emit(
            {
                "leases": [
                    {**lease, "expired": store.is_expired(lease)}
                    for lease in leases
                ]
            }
        )
        return 0
    except (GPULeaseError, ValueError) as exc:
        _emit({"ok": False, "error": str(exc), "error_type": type(exc).__name__})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
