from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from src.files import atomic_write_json, process_file_lock
from src.paths import SOURCE_ROOT, dataset_folder_key, normalize_experiment_name


def run_dataset_batch(data_dirs: list[Path], *, argv: list[str], output_dir: Path, max_workers: int,
                      experiment_name: str | None = None, reuse_outputs: bool = False):
    """Run the same config for each folder in bounded, independent Python processes."""
    if max_workers < 1:
        raise ValueError("run.max_parallel_datasets must be positive")
    paths = [path.resolve() for path in data_dirs]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Dataset folders must be non-empty and distinct")
    resolved_name = normalize_experiment_name(experiment_name, default="experiment")
    batch = output_dir / resolved_name
    log_dir = batch / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for path in paths:
        key = dataset_folder_key(path)
        command = [sys.executable, str(SOURCE_ROOT / "main.py"), *argv,
                   "--data-dir", str(path)]
        if experiment_name:
            command.extend(["--experiment-name", resolved_name])
        command.extend(["--output-folder", str(output_dir.resolve())])
        jobs.append({"data_dir": str(path), "command": command, "log": str(log_dir / f"{key}.log")})
    manifest = {"status": "running", "max_workers": max_workers,
                "output_folder": str(output_dir.resolve()), "experiment_name": resolved_name,
                "experiment_dir": str(batch.resolve()), "reuse_outputs": reuse_outputs,
                "started_at": datetime.now(timezone.utc).isoformat(), "jobs": jobs}
    manifest_path = batch / "batch_manifest.json"

    def save():
        atomic_write_json(manifest_path, manifest)

    def execute(job):
        print(f"Starting {job['data_dir']}; log: {job['log']}", flush=True)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        with Path(job["log"]).open("w", encoding="utf-8") as log:
            result = subprocess.run(job["command"], stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
        return result.returncode

    with process_file_lock(batch / ".mapf-batch.lock", blocking=False,
                           conflict_message=f"Another dataset batch is writing to {batch}; use a different experiment_name"):
        save()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            pending = {pool.submit(execute, job): job for job in jobs}
            for future in as_completed(pending):
                job = pending[future]
                try:
                    job["returncode"] = future.result()
                except Exception as exc:
                    job["returncode"], job["error"] = 1, str(exc)
                print(f"Finished {job['data_dir']}: exit={job['returncode']}", flush=True)
                save()
        manifest["status"] = "success" if all(job["returncode"] == 0 for job in jobs) else "failed"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        save()
        if manifest["status"] == "failed":
            raise RuntimeError(f"One or more dataset runs failed; see {manifest_path} and the individual logs")
    return {"batch_dir": batch, "batch_manifest_json": manifest_path}
