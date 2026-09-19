"""Run independent dataset training jobs in separate, bounded child processes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from dataset_profiles import CITIES, normalize_dataset


def dataset_job(value):
    parts = value.split(":")
    if len(parts) > 2:
        raise ValueError(f"Invalid dataset selector: {value}")
    dataset = normalize_dataset(parts[0])
    city = parts[1].upper() if len(parts) == 2 else "JHB" if dataset == "charged" else None
    if city and (dataset != "charged" or city not in CITIES):
        raise ValueError(f"Invalid city in {value}")
    return dataset, city


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["urbanev", "charged:JHB", "mp_evdata"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+", default=["AR"], choices=["AR", "lstm", "chronos", "timesfm"])
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--output-folder", type=Path, default=Path("output/parallel"))
    parser.add_argument("--pipeline-stage", choices=["forecaster", "full"], default="forecaster")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--lstm-epochs", type=int)
    parser.add_argument("--horizon-days", type=int, default=2)
    parser.add_argument("--diurnal-blend-alpha", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    selections = [dataset_job(value) for value in args.datasets]
    if len(set(selections)) != len(selections):
        parser.error("Duplicate dataset/city jobs are not allowed")
    batch = args.output_folder / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    batch.mkdir(parents=True)
    jobs = []
    for dataset, city in selections:
        key = dataset + ("_" + city if city else "")
        command = [sys.executable, str(Path(__file__).resolve().with_name("main.py")),
                   "--config", args.config, "--dataset", dataset,
                   "--forecast-models", *args.models, "--pipeline-stage", args.pipeline_stage,
                   "--output-folder", str(batch), "--horizon-days", str(args.horizon_days),
                   "--diurnal-blend-alpha", str(args.diurnal_blend_alpha)]
        if city:
            command.extend(["--city", city])
        if args.dry_run:
            command.append("--dry-run")
        if args.device:
            command.extend(["--device", args.device])
        if args.lstm_epochs is not None:
            command.extend(["--lstm-epochs", str(args.lstm_epochs)])
        jobs.append({"dataset": key, "command": command, "log": str(batch / f"{key}.log")})
    manifest = {"status": "running", "max_workers": args.max_workers, "jobs": jobs}
    manifest_path = batch / "batch_manifest.json"
    def save():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)
    save()

    def execute(job):
        print(f"Starting {job['dataset']}; log: {job['log']}", flush=True)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        with Path(job["log"]).open("w", encoding="utf-8") as log:
            result = subprocess.run(job["command"], stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
        return result.returncode

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        pending = {pool.submit(execute, job): job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            try:
                job["returncode"] = future.result()
            except Exception as exc:
                job["returncode"], job["error"] = 1, str(exc)
            print(f"Finished {job['dataset']}: exit={job['returncode']}", flush=True)
            save()
    manifest["status"] = "success" if all(job["returncode"] == 0 for job in jobs) else "failed"
    save()
    print(f"Batch {manifest['status']}: {manifest_path}")
    return 0 if manifest["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
