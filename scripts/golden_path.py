"""Golden path: run the whole project end to end from a clean state.

Stages (each runs as its own process with the current interpreter):

    1. tests       offline unit/integration tests
    2. pipeline    download, clean, aggregate, features, splits
    3. train       tune + train + register both models in MLflow
    4. batch       recursive batch demand forecast -> predictions.db
    5. serve       start the API in-process, check /health and one prediction
    6. responsible fairness + SHAP + model cards
    7. drift       drift report for both problems

Every stage's exit code, duration and the key outputs are written to
``artifacts/golden_path/run_summary.json`` and the full log to
``artifacts/golden_path/run.log``. The run stops at the first failing stage.

Usage::

    python -m scripts.golden_path --clean            # fresh start, full run
    python -m scripts.golden_path --skip-download    # reuse cached raw data
    python -m scripts.golden_path --n-trials 25      # full tuning budget
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "golden_path"

# Generated outputs removed by --clean. Raw downloads are kept unless
# --clean-raw is also given, so a clean run does not always re-download.
GENERATED = [
    "data/processed",
    "data/features",
    "mlruns",
    "mlflow.db",
    "predictions.db",
    "artifacts",
    "docs/model_cards/demand.md",
    "docs/model_cards/fare.md",
]

SERVE_SMOKE = """
import json
from fastapi.testclient import TestClient
from src.serving.api import app

with TestClient(app) as client:
    health = client.get("/health").json()
    print("health:", json.dumps(health))
    assert health["status"] == "ok", health
    fare = client.post("/predict/fare", json={
        "trip_distance": 3.4, "pu_location_id": 132, "do_location_id": 230,
        "passenger_count": 1, "pickup_datetime": "2024-01-15T18:30:00",
    })
    print("predict/fare:", fare.status_code, fare.text)
    assert fare.status_code == 200, fare.text
"""


def _stages(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    py = sys.executable
    pipeline = [py, "-m", "scripts.run_pipeline", "--year", str(args.year), "--month", str(args.month)]
    if args.skip_download:
        pipeline.append("--skip-download")
    return [
        ("tests", [py, "-m", "pytest", "-q"]),
        ("pipeline", pipeline),
        ("train", [py, "-m", "scripts.train_models", "--n-trials", str(args.n_trials)]),
        ("batch", [py, "-m", "scripts.batch_forecast", "--horizon", str(args.horizon)]),
        ("serve", [py, "-c", SERVE_SMOKE]),
        ("responsible", [py, "-m", "scripts.responsible.run_analysis"]),
        ("drift", [py, "-m", "scripts.monitoring.demo_drift_report", "--problem", "all", "--mode", "drift"]),
    ]


def _clean(include_raw: bool) -> None:
    paths = GENERATED + (["data/raw"] if include_raw else [])
    for rel in paths:
        target = ROOT / rel
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        else:
            continue
        print(f"removed {rel}")


def _outputs() -> dict[str, object]:
    """Collect the evidence each stage is expected to leave behind."""
    found: dict[str, object] = {}
    summary = ROOT / "data" / "processed" / "pipeline_summary.json"
    if summary.exists():
        found["pipeline_summary"] = json.loads(summary.read_text(encoding="utf-8"))
    for problem in ("demand", "fare"):
        card = ROOT / "artifacts" / problem / f"{problem}_model_card.md"
        fairness = ROOT / "artifacts" / "responsible" / f"{problem}_fairness.json"
        drift = ROOT / "artifacts" / "drift" / f"{problem}_drift_metrics.json"
        found[problem] = {
            "model_card": card.exists(),
            "fairness_report": fairness.exists(),
            "drift_report": drift.exists(),
        }
    found["predictions_db"] = (ROOT / "predictions.db").exists()
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the project end to end.")
    parser.add_argument("--clean", action="store_true", help="Delete generated outputs first.")
    parser.add_argument("--clean-raw", action="store_true", help="With --clean, also delete raw downloads.")
    parser.add_argument("--skip-download", action="store_true", help="Reuse cached raw data.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--n-trials", type=int, default=1, help="Optuna trials per estimator.")
    parser.add_argument("--horizon", type=int, default=3, help="Batch forecast hours.")
    args = parser.parse_args()

    if args.clean:
        _clean(args.clean_raw)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "run.log"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    run = {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit or None,
        "args": vars(args),
        "stages": [],
    }

    ok = True
    with log_path.open("w", encoding="utf-8") as log:
        for name, cmd in _stages(args):
            banner = f"\n===== [{name}] {' '.join(cmd[:4])} ====="
            print(banner, flush=True)
            log.write(banner + "\n")
            log.flush()
            start = time.monotonic()
            proc = subprocess.Popen(
                cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            code = proc.wait()
            seconds = round(time.monotonic() - start, 1)
            status = "passed" if code == 0 else "failed"
            run["stages"].append({"stage": name, "status": status, "exit_code": code, "seconds": seconds})
            print(f"----- [{name}] {status} in {seconds}s", flush=True)
            if code != 0:
                ok = False
                break

    run["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    run["result"] = "passed" if ok else "failed"
    run["outputs"] = _outputs()
    (OUT_DIR / "run_summary.json").write_text(json.dumps(run, indent=2), encoding="utf-8")

    print("\n===== Golden path summary =====")
    for stage in run["stages"]:
        print(f"  {stage['stage']:<12} {stage['status']:<7} {stage['seconds']:>7}s")
    print(f"  result: {run['result']}  commit: {commit[:7] if commit else 'n/a'}")
    print(f"  summary: {OUT_DIR / 'run_summary.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
