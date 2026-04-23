from __future__ import annotations

from pathlib import Path

try:
    from hf_text_summary.cli import main
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).parent / "src"))
    from hf_text_summary.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
