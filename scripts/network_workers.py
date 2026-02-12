#!/usr/bin/env python3
"""
Scan local network workers and correlate with heartbeat files.

Reads worker heartbeats from:
  - shared/gpus/gpu_*/heartbeat.json
  - shared/cpus/*/heartbeat.json
  - shared/heartbeats/*.json (unified mirror, optional)
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text())


def resolve_shared_path(config_path: Path, config: dict) -> Path:
    shared = Path(config.get("shared_path", "../"))
    if shared.is_absolute():
        return shared
    return (config_path.resolve().parent / shared).resolve()


def heartbeat_age(last_updated: str) -> int | None:
    if not last_updated:
        return None
    try:
        return int((datetime.now() - datetime.fromisoformat(last_updated)).total_seconds())
    except Exception:
        return None


def load_heartbeats(shared_path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}

    for hb_file in sorted((shared_path / "gpus").glob("gpu_*/heartbeat.json")):
        try:
            hb = json.loads(hb_file.read_text())
            name = hb.get("name", hb_file.parent.name)
            rows[name] = {
                "name": name,
                "worker_type": "gpu",
                "hostname": hb.get("hostname"),
                "ip_address": hb.get("ip_address"),
                "state": hb.get("state"),
                "last_updated": hb.get("last_updated"),
                "age_s": heartbeat_age(hb.get("last_updated", "")),
                "path": str(hb_file),
            }
        except Exception:
            continue

    for hb_file in sorted((shared_path / "cpus").glob("*/heartbeat.json")):
        try:
            hb = json.loads(hb_file.read_text())
            name = hb.get("name", hb_file.parent.name)
            rows[name] = {
                "name": name,
                "worker_type": "cpu",
                "hostname": hb.get("hostname"),
                "ip_address": hb.get("ip_address"),
                "state": hb.get("state"),
                "last_updated": hb.get("last_updated"),
                "age_s": heartbeat_age(hb.get("last_updated", "")),
                "path": str(hb_file),
            }
        except Exception:
            continue

    unified = shared_path / "heartbeats"
    if unified.exists():
        for hb_file in sorted(unified.glob("*.json")):
            try:
                hb = json.loads(hb_file.read_text())
                name = hb.get("name", hb_file.stem)
                if name in rows:
                    continue
                rows[name] = {
                    "name": name,
                    "worker_type": hb.get("worker_type", "unknown"),
                    "hostname": hb.get("hostname"),
                    "ip_address": hb.get("ip_address"),
                    "state": hb.get("state"),
                    "last_updated": hb.get("last_updated"),
                    "age_s": heartbeat_age(hb.get("last_updated", "")),
                    "path": str(hb_file),
                }
            except Exception:
                continue

    return rows


def get_default_subnet() -> str | None:
    try:
        out = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "scope", "global"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except Exception:
        return None

    cidrs: List[str] = []
    for line in out.splitlines():
        m = re.search(r"\binet\s+([0-9.]+/[0-9]+)\b", line)
        if m:
            cidr = m.group(1)
            if not cidr.startswith("127."):
                cidrs.append(cidr)

    if not cidrs:
        return None

    for cidr in cidrs:
        try:
            net = ipaddress.ip_interface(cidr).network
            if net.is_private:
                return str(net)
        except Exception:
            continue
    return str(ipaddress.ip_interface(cidrs[0]).network)


def parse_ip_neigh() -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    try:
        out = subprocess.run(["ip", "neigh"], capture_output=True, text=True, check=False).stdout
    except Exception:
        return rows

    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        ip = parts[0]
        mac = None
        state = parts[-1]
        if "lladdr" in parts:
            idx = parts.index("lladdr")
            if idx + 1 < len(parts):
                mac = parts[idx + 1]
        rows[ip] = {"ip": ip, "mac": mac, "neigh_state": state}
    return rows


def _ping_one(ip: str) -> tuple[str, bool]:
    rc = subprocess.run(
        ["ping", "-c", "1", "-W", "1", ip],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode
    return ip, (rc == 0)


def ping_sweep(subnet: str, max_workers: int = 48) -> List[str]:
    net = ipaddress.ip_network(subnet, strict=False)
    ips = [str(ip) for ip in net.hosts()]
    alive: List[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_ping_one, ip) for ip in ips]
        for fut in as_completed(futures):
            ip, ok = fut.result()
            if ok:
                alive.append(ip)
    return sorted(alive)


def reverse_dns(ip: str) -> str | None:
    try:
        host, _, _ = socket.gethostbyaddr(ip)
        return host
    except Exception:
        return None


def build_rows(heartbeats: Dict[str, dict], hosts: Dict[str, dict]) -> List[dict]:
    by_ip = {}
    by_hostname = {}
    for hb in heartbeats.values():
        if hb.get("ip_address"):
            by_ip[hb["ip_address"]] = hb
        if hb.get("hostname"):
            by_hostname[str(hb["hostname"]).lower()] = hb

    rows: List[dict] = []
    matched_hb = set()
    for ip, host in sorted(hosts.items()):
        dns = host.get("dns")
        hb = by_ip.get(ip)
        if hb is None and dns:
            hb = by_hostname.get(dns.lower())
        if hb:
            matched_hb.add(hb["name"])
        rows.append({
            "ip": ip,
            "dns": dns,
            "mac": host.get("mac"),
            "neigh_state": host.get("neigh_state"),
            "heartbeat": hb,
        })

    # Unmatched heartbeat-only workers
    for hb in heartbeats.values():
        if hb["name"] not in matched_hb:
            rows.append({
                "ip": hb.get("ip_address"),
                "dns": hb.get("hostname"),
                "mac": None,
                "neigh_state": None,
                "heartbeat": hb,
            })

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan network workers + heartbeat correlation")
    default_config = str(Path(__file__).resolve().parent.parent / "shared" / "agents" / "config.json")
    parser.add_argument("--config", default=default_config, help="Path to config.json")
    parser.add_argument("--subnet", default=None, help="CIDR subnet (e.g. 10.0.0.0/24)")
    parser.add_argument("--sweep", action="store_true", help="Ping sweep the subnet")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    shared_path = resolve_shared_path(config_path, config)
    heartbeats = load_heartbeats(shared_path)

    hosts = parse_ip_neigh()
    subnet = args.subnet or get_default_subnet()
    if args.sweep and subnet:
        alive = ping_sweep(subnet)
        hosts = parse_ip_neigh()
        for ip in alive:
            hosts.setdefault(ip, {"ip": ip, "mac": None, "neigh_state": "REACHABLE"})

    for ip in list(hosts.keys()):
        hosts[ip]["dns"] = reverse_dns(ip)

    rows = build_rows(heartbeats, hosts)

    if args.json:
        print(json.dumps({
            "subnet": subnet,
            "host_count": len(hosts),
            "heartbeat_count": len(heartbeats),
            "rows": rows,
        }, indent=2))
        return 0

    print("=" * 84)
    print("NETWORK WORKER SCAN")
    print("=" * 84)
    print(f"Subnet: {subnet or '(auto-detect failed)'}")
    print(f"Hosts seen: {len(hosts)} | Heartbeats: {len(heartbeats)}")
    print()
    print(f"{'IP':15} {'DNS':24} {'NEIGH':10} {'HB_NAME':14} {'TYPE':4} {'STATE':8} {'AGE':6}")
    print("-" * 84)
    for r in rows:
        hb = r.get("heartbeat")
        hb_name = hb.get("name") if hb else "-"
        hb_type = hb.get("worker_type") if hb else "-"
        hb_state = hb.get("state") if hb else "-"
        hb_age = f"{hb.get('age_s')}s" if hb and hb.get("age_s") is not None else "-"
        ip = (r.get("ip") or "-")[:15]
        dns = (r.get("dns") or "-")[:24]
        neigh = (r.get("neigh_state") or "-")[:10]
        print(f"{ip:15} {dns:24} {neigh:10} {hb_name[:14]:14} {hb_type[:4]:4} {hb_state[:8]:8} {hb_age:6}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
