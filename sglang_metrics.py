"""Best-effort, run-scoped snapshots of server-wide SGLang Prometheus metrics."""
from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx


_LOG = logging.getLogger(__name__)
_SAMPLE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{((?:[^{}"\\]|"(?:\\.|[^"\\])*")*)\})?'
    r'\s+(\S+)(?:\s+[^\s#]+)?(?:\s*#.*)?$'
)
_LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:\\.|[^"\\])*)"\s*(?:,|$)')
_TYPE = re.compile(r'^#\s+TYPE\s+(\S+)\s+(\S+)\s*$')
_CSV_FIELDS = (
    "scrape_index", "phase", "timestamp", "name", "labels", "type", "component", "value", "value_text",
)
_SUMMARY_FIELDS = (
    "name", "labels", "type", "component", "sample_count", "observation_count", "first_value",
    "last_value", "final_value", "mean", "min", "max", "delta", "known_observed_delta", "mean_per_second",
    "reset_count", "window_complete", "histogram_count", "histogram_sum", "histogram_mean",
)


def _origin(url: str) -> tuple[str, str | None, int | None]:
    parsed = urlsplit(url)
    return parsed.scheme.lower(), parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


def resolve_metrics_url(base_url: str, override: str | None = None) -> str:
    """Replace only a trailing /v1, preserving any reverse-proxy path prefix."""
    parsed = urlsplit(override if override else base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("SGLang metrics requires an HTTP(S) URL without embedded credentials")
    if override:
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, path + "/metrics", "", ""))


def _labels(source: str) -> dict[str, str]:
    def unescape(match: re.Match) -> str:
        char = match[1]
        if char not in {'\\', '"', 'n'}:
            raise ValueError("Invalid Prometheus label escape")
        return "\n" if char == "n" else char

    labels: dict[str, str] = {}
    position = 0
    while source[position:].strip():
        match = _LABEL.match(source, position)
        if match is None or match[1] in labels:
            raise ValueError("Invalid Prometheus labels")
        labels[match[1]] = re.sub(r'\\(.)', unescape, match[2])
        position = match.end()
    return labels


def parse_prometheus_text(text: str) -> list[dict[str, Any]]:
    """Read SGLang samples; retain labels and nonfinite values without guessing zero."""
    types = {match[1]: match[2] for line in text.splitlines() if (match := _TYPE.match(line.strip()))}
    samples: list[dict[str, Any]] = []
    seen = set()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("sglang:"):
            continue
        match = _SAMPLE.match(line)
        if match is None:
            raise ValueError("Invalid SGLang metric sample")
        name, label_text, raw_value = match.groups()
        labels = _labels(label_text or "")
        key = name, json.dumps(labels, sort_keys=True, ensure_ascii=False)
        if key in seen:
            raise ValueError("Duplicate SGLang metric series")
        seen.add(key)
        value = float(raw_value)
        family, component = name, ""
        metric_type = types.get(name, "untyped")
        if metric_type == "untyped":
            for suffix in ("_bucket", "_sum", "_count", "_created", "_total"):
                candidate = name[:-len(suffix)] if name.endswith(suffix) else ""
                if candidate in types:
                    family, component, metric_type = candidate, suffix[1:], types[candidate]
                    break
        if component == "created":
            metric_type = "gauge"
        samples.append({"name": name, "labels": labels, "type": metric_type,
                        "family": family, "component": component,
                        "value": value if math.isfinite(value) else None, "value_text": raw_value})
    return samples


class SGLangMetricsCollector:
    """Sample a server during one Agent run; never attribute server totals to calls."""

    def __init__(self, *, base_url: str, api_key: str | None, output_dir: Path,
                 settings: dict[str, Any], context: dict[str, Any]) -> None:
        self.base_url, self.api_key = base_url, api_key
        self.settings = dict(settings)
        self.directory = Path(output_dir)
        self.context = {key: value for key, value in context.items() if key in {
            "dataset", "dataset_adapter", "dataset_fingerprint", "model", "profile", "agent_mode",
            "forecast_origin", "forecast_backend", "selected_zone_ids", "agent_model", "agent_profile", "zone_ids",
        }}
        self._client: httpx.AsyncClient | None = None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._started = self._stopped = self._final_success = False
        self._scrape_count = self._success_count = 0
        self._series: dict[tuple[str, str], dict[str, Any]] = {}
        self._errors: dict[str, int] = {}
        self._started_at: str | None = None
        self._started_monotonic = 0.0
        self._written_outputs: dict[str, Path] = {}

    @property
    def outputs(self) -> dict[str, Path]:
        return {name: path for name, path in self._written_outputs.items() if path.exists()}

    def _error(self, code: str) -> None:
        self._errors[code] = self._errors.get(code, 0) + 1

    async def start(self) -> None:
        if self._started or not self.settings.get("enabled", True):
            return
        self._started = True
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._started_monotonic = time.monotonic()
        try:
            (self.directory / "snapshots").mkdir(parents=True, exist_ok=True)
            # Replace only this collector's filenames; never recursively remove
            # directories or touch unrelated files in the output directory.
            for name in ("summary.json", "summary.csv"):
                (self.directory / name).unlink(missing_ok=True)
            for path in (self.directory / "snapshots").iterdir():
                if path.is_file() and re.fullmatch(r"[0-9]{6,}_(initial|periodic|final)\.prom", path.name):
                    path.unlink()
            (self.directory / "scrapes.jsonl").write_text("", encoding="utf-8")
            self._written_outputs["sglang_metrics_dir"] = self.directory
            self._written_outputs["sglang_metrics_scrapes_jsonl"] = self.directory / "scrapes.jsonl"
            with (self.directory / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=_CSV_FIELDS).writeheader()
            self._written_outputs["sglang_metrics_samples_csv"] = self.directory / "samples.csv"
            self._url = resolve_metrics_url(self.base_url, self.settings.get("url"))
            headers = {}
            if self.api_key and self.api_key != "EMPTY" and _origin(self._url) == _origin(self.base_url):
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(timeout=float(self.settings.get("timeout_seconds", 3.0)),
                                             headers=headers, trust_env=False, follow_redirects=False)
            await self._capture("initial")
            self._task = asyncio.create_task(self._poll())
        except Exception as exc:
            self._error("setup_" + type(exc).__name__)
            _LOG.warning("SGLang metrics setup failed; inference will continue.")

    async def _poll(self) -> None:
        interval = float(self.settings.get("interval_seconds", 5.0))
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                if not self._stop.is_set():
                    await self._capture("periodic")

    async def _capture(self, phase: str) -> None:
        self._scrape_count += 1
        index = self._scrape_count
        record: dict[str, Any] = {"scrape_index": index, "phase": phase,
                                  "timestamp": datetime.now(timezone.utc).isoformat(), "status": "error"}
        try:
            response = await self._client.get(self._url)
            record["http_status"] = response.status_code
            if response.status_code != 200:
                record["error"] = f"http_{response.status_code}"
            else:
                snapshot = Path("snapshots") / f"{index:06d}_{phase}.prom"
                (self.directory / snapshot).write_text(response.text, encoding="utf-8")
                record["snapshot"] = snapshot.as_posix()
                samples = parse_prometheus_text(response.text)
                record["sample_count"] = len(samples)
                if not samples:
                    record["error"] = "no_sglang_samples"
                else:
                    with (self.directory / "samples.csv").open("a", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
                        for sample in samples:
                            row = {key: sample[key] for key in ("name", "type", "component", "value", "value_text")}
                            row.update({"labels": json.dumps(sample["labels"], sort_keys=True, ensure_ascii=False),
                                        "scrape_index": index, "phase": phase, "timestamp": record["timestamp"]})
                            writer.writerow(row)
                    observed_at = time.monotonic()
                    for sample in samples:
                        self._observe(sample, index, observed_at)
                    record["status"] = "ok"
                    self._success_count += 1
                    self._final_success = phase == "final"
        except Exception as exc:
            # Exception strings and response bodies may include URL credentials.
            record["error"] = type(exc).__name__
        if "error" in record:
            self._error(record["error"])
        try:
            with (self.directory / "scrapes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        except Exception as exc:
            self._error("scrape_log_" + type(exc).__name__)

    def _observe(self, sample: dict[str, Any], index: int, observed_at: float) -> None:
        key = sample["name"], json.dumps(sample["labels"], sort_keys=True, ensure_ascii=False)
        state = self._series.setdefault(key, {
            "name": sample["name"], "labels": sample["labels"], "type": sample["type"],
            "family": sample["family"], "component": sample["component"], "sample_count": 0,
            "observation_count": 0, "first_index": index, "first_value": sample["value"],
            "first_time": observed_at, "last_time": observed_at, "last_index": 0, "last_value": None,
            "min": None, "max": None, "mean": 0.0, "known_observed_delta": 0.0, "reset_count": 0,
            "delta_pair_count": 0,
            "type_consistent": True,
        })
        value = sample["value"]
        state["observation_count"] += 1
        state["type_consistent"] &= state["type"] == sample["type"]
        if value is not None:
            previous = state["last_value"]
            if previous is not None and self._cumulative(state):
                if value < previous:
                    state["reset_count"] += 1
                elif state["last_index"] == index - 1:
                    state["known_observed_delta"] += value - previous
                    state["delta_pair_count"] += 1
            state["sample_count"] += 1
            count = state["sample_count"]
            state["mean"] = state["mean"] * ((count - 1) / count) + value / count
            state["min"] = value if state["min"] is None else min(state["min"], value)
            state["max"] = value if state["max"] is None else max(state["max"], value)
        state.update(last_value=value, last_index=index, last_time=observed_at)

    @staticmethod
    def _cumulative(state: dict[str, Any]) -> bool:
        return state["type"] == "counter" or (
            state["type"] in {"histogram", "summary"} and state["component"] in {"sum", "count", "bucket"})

    def _summary_rows(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        series = []
        for key in sorted(self._series):
            state = self._series[key]
            complete = bool(self._final_success and self._scrape_count >= 2 and state["first_index"] == 1
                            and state["last_index"] == self._scrape_count
                            and state["sample_count"] == self._scrape_count and state["type_consistent"]
                            and not state["reset_count"] and not self._errors)
            cumulative = self._cumulative(state)
            elapsed = state["last_time"] - state["first_time"]
            delta = state["last_value"] - state["first_value"] if cumulative and complete else None
            row = {name: state[name] for name in (
                "name", "labels", "type", "component", "sample_count", "observation_count", "first_value",
                "last_value", "reset_count",
            )}
            row.update({"window_complete": complete, "delta": delta,
                        "final_value": state["last_value"] if self._final_success and state["last_index"] == self._scrape_count else None,
                        "known_observed_delta": state["known_observed_delta"] if cumulative and state["delta_pair_count"] else None,
                        "mean_per_second": delta / elapsed if delta is not None and elapsed > 0 else None,
                        "mean": state["mean"] if not cumulative and state["sample_count"] else None,
                        "min": state["min"] if not cumulative else None,
                        "max": state["max"] if not cumulative else None})
            series.append(row)
        index = {(row["name"], json.dumps(row["labels"], sort_keys=True)): row for row in series}
        histograms = []
        for state in self._series.values():
            if state["type"] != "histogram" or state["component"] != "count":
                continue
            labels = json.dumps(state["labels"], sort_keys=True)
            count = index[(state["name"], labels)]
            total = index.get((state["family"] + "_sum", labels))
            complete = bool(count["window_complete"] and total and total["window_complete"])
            delta_count = count["delta"] if complete else None
            delta_sum = total["delta"] if complete else None
            histograms.append({"name": state["family"], "labels": state["labels"], "type": "histogram_summary",
                               "window_complete": complete, "histogram_count": delta_count, "histogram_sum": delta_sum,
                               "histogram_mean": delta_sum / delta_count if delta_count else None})
        return series, histograms

    def _write_summary(self) -> None:
        series, histograms = self._summary_rows()
        payload = {
            "schema_version": 1, "scope": "server", "backend": "sglang", "context": self.context,
            "started_at": self._started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": max(0.0, time.monotonic() - self._started_monotonic),
            "endpoint_source": "explicit" if self.settings.get("url") else "derived_from_base_url",
            "interval_seconds": self.settings.get("interval_seconds", 5.0),
            "scrape_count": self._scrape_count, "successful_scrape_count": self._success_count,
            "scrape_coverage": self._success_count / self._scrape_count if self._scrape_count else None,
            "complete_series_count": sum(row["window_complete"] for row in series),
            "incomplete_series_count": sum(not row["window_complete"] for row in series),
            "status": "unavailable" if not self._success_count else "partial" if (
                self._errors or any(not row["window_complete"] for row in series)) else "ok",
            "errors": self._errors, "series": series, "histograms": histograms,
            "notes": [
                "Server-wide observation window; concurrent runs or other clients may contribute. Not per-run billed usage.",
                "Series retain all worker/model/source labels; do not add overlapping total and component series.",
                "Counter delta requires initial and final values, all finite observations, and no observed reset or scrape error.",
                "known_observed_delta includes only nondecreasing adjacent observed pairs; it is not a complete total.",
                "Gauge mean is a sampled arithmetic mean, not a time-weighted mean or a token-weighted cache-hit ratio.",
                "last_value is the last observed value; final_value is unknown if the final scrape or series is unavailable.",
                "cache_hit_rate is a server gauge; it does not replace call-level cached_tokens or token accounting.",
                "token_usage measures KV/state-pool occupancy, not GPU compute utilization.",
                "Histogram means use observed sum/count deltas per complete label set; buckets remain in samples for later analysis.",
                "Metrics must be enabled on the SGLang server with --enable-metrics; absent metrics remain unknown.",
                "Periodic collection is best effort; synchronous work on the event loop can lengthen sampling intervals.",
            ],
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        self._written_outputs["sglang_metrics_summary_json"] = self.directory / "summary.json"
        with (self.directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=_SUMMARY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in series + histograms:
                writer.writerow({**row, "labels": json.dumps(row["labels"], sort_keys=True, ensure_ascii=False)})
        self._written_outputs["sglang_metrics_summary_csv"] = self.directory / "summary.csv"

    async def stop(self) -> None:
        if not self._started or self._stopped:
            return
        self._stopped = True
        self._stop.set()
        try:
            if self._task is not None:
                await self._task
            if self._client is not None:
                await self._capture("final")
        except Exception as exc:
            self._error("stop_" + type(exc).__name__)
        finally:
            if self._client is not None:
                try:
                    await self._client.aclose()
                except Exception as exc:
                    self._error("close_" + type(exc).__name__)
            try:
                self._write_summary()
            except Exception as exc:
                self._error("summary_" + type(exc).__name__)
                _LOG.warning("SGLang metrics summary could not be saved; inference results are unaffected.")
