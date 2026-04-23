"""Compatibility shim.

The production code lives under `src/hf_text_summary/`.
This module keeps the original import path (`import summarizer`) working.
"""

from pathlib import Path

try:
	from hf_text_summary import *  # type: ignore  # noqa: F403
except ModuleNotFoundError:
	import sys

	sys.path.append(str(Path(__file__).parent / "src"))
	from hf_text_summary import *  # type: ignore  # noqa: F403
