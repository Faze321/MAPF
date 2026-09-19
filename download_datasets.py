"""Download the official hourly inputs, preserving source versions and checksums."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
import uuid

CITIES = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")
CHARGED_FILES = ("volume.csv", "e_price.csv", "s_price.csv", "weather.csv", "sites.csv", "poi.csv", "info.csv")
MP_FILES = ("station-level load Profile 1h.xlsx", "price.xlsx", "README.md")


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "MAPF-dataset-importer"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def checksum(path, algorithm, size):
    digest = hashlib.new("sha1" if algorithm == "git" else algorithm)
    if algorithm == "git":
        digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(entry):
    path = Path(entry["path"])
    size = entry["size"]
    if path.is_file() and path.stat().st_size == size and checksum(path, entry["algorithm"], size) == entry["checksum"]:
        print(f"Verified {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.part")
    try:
        request = urllib.request.Request(entry["url"], headers={"User-Agent": "MAPF-dataset-importer"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(chunk)
        if temporary.stat().st_size != size or checksum(temporary, entry["algorithm"], size) != entry["checksum"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        temporary.replace(path)
        print(f"Downloaded {path}", flush=True)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", choices=["charged", "mp_evdata"], default=["charged", "mp_evdata"])
    parser.add_argument("--cities", nargs="+", choices=CITIES, default=list(CITIES))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    for dataset in args.datasets:
        entries = []
        if dataset == "charged":
            root = args.data_root / "CHARGED"
            commit = get_json("https://api.github.com/repos/IntelligentSystemsLab/CHARGED/commits/main")["sha"]
            tree = get_json(f"https://api.github.com/repos/IntelligentSystemsLab/CHARGED/git/trees/{commit}?recursive=1")
            if tree.get("truncated"):
                raise ValueError("Incomplete CHARGED source tree")
            blobs = {item["path"]: item for item in tree["tree"]}
            for city in args.cities:
                for filename in CHARGED_FILES:
                    source = f"data/{city}_remove_zero/{filename}"
                    blob = blobs[source]
                    entries.append(dict(path=str(root / city / filename), size=blob["size"],
                                        algorithm="git", checksum=blob["sha"],
                                        url=f"https://raw.githubusercontent.com/IntelligentSystemsLab/CHARGED/{commit}/{source}"))
            metadata = {"dataset": dataset, "commit": commit, "variant": "remove_zero", "files": entries,
                        "source": "https://github.com/IntelligentSystemsLab/CHARGED", "cities": args.cities}
        else:
            root = args.data_root / "MP-EVData"
            source = get_json("https://api.figshare.com/v2/articles/29882366")
            # Resolve the current release to immutable file IDs; do not execute upstream code.
            files = {file["name"]: file for file in source["files"]}
            for filename in MP_FILES:
                file = files[filename]
                entries.append(dict(path=str(root / filename), size=file["size"], algorithm="md5",
                                    checksum=file["supplied_md5"], url=file["download_url"]))
            metadata = {"dataset": dataset, "doi": source["doi"], "license": source["license"], "files": entries}
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(download, entries))
        (root / "download_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
