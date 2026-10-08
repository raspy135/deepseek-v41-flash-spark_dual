"""Arm/read/profile/stop the loaded engine; print a compact latency census."""
import argparse
from fnmatch import fnmatchcase
import json
from pathlib import Path
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("status", "arm", "run", "stop", "show"))
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--names", nargs="+")
    p.add_argument("--max-cases", type=int)
    p.add_argument("--snapshot-budget-mb", type=int)
    p.add_argument("--warmup", type=int)
    p.add_argument("--repeats", type=int)
    p.add_argument("--calls", type=int)
    p.add_argument("--flush-mb", type=int)
    p.add_argument("--out", type=Path)
    p.add_argument("--input", type=Path, help="read a saved report with action=show")
    p.add_argument("--top", type=int, default=15)
    a = p.parse_args()
    if a.action == "show":
        if a.input is None:
            p.error("show requires --input")
        report = json.loads(a.input.read_text())
    else:
        url = a.url.rstrip("/") + "/v1/decode-probe"
        body = {"action": a.action}
        for key in ("names", "max_cases", "snapshot_budget_mb", "warmup", "repeats", "calls", "flush_mb"):
            if getattr(a, key) is not None:
                body[key] = getattr(a, key)
        req = urllib.request.Request(url) if a.action == "status" else urllib.request.Request(
            url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                report = json.load(r)
        except urllib.error.HTTPError as exc:
            p.exit(1, f"HTTP {exc.code}: {exc.read().decode(errors='replace')}\n")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"action={report.get('action')} ok={report.get('ok')}")
    for rank in report.get("ranks", ()):
        r = rank.get("report", rank)
        print(f"rank {rank['rank']}: {r.get('status', rank.get('error'))} {r.get('coverage', {})}")
        rows = [c for c in r.get("cases", ()) if c.get("live_ms") is not None]
        if a.names:
            rows = [c for c in rows if any(fnmatchcase(c["name"], pattern) for pattern in a.names)]
        rows.sort(key=lambda c: c["live_ms"], reverse=True)
        print(" live_ms  warm_ms  cold_ms  exact  operation (nested spans overlap)")
        for c in rows[:a.top]:
            iso = c.get("isolation", {})
            def fmt(v):
                return f"{v:8.3f}" if isinstance(v, (int, float)) else "       -"
            exact = "stale" if iso.get("stale") else str(iso.get("exact", "-"))
            print(f"{fmt(c['live_ms'])} {fmt(iso.get('warm_ms'))} {fmt(iso.get('cold_ms'))} {exact:>6} {c['name']}")
        for phase in r.get("loop_phases", {}).get("phases", ()):
            print(f"  {phase['name']}: allocations={phase['allocation_requests']} "
                  f"device_allocations={phase['device_allocations']} "
                  f"capture_intervals={phase['capture_intervals']}")
    if not report.get("ok", True):
        p.exit(1)


if __name__ == "__main__":
    main()
