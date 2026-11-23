if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.cli import cli


if __name__ == "__main__":
    raise SystemExit(cli())
