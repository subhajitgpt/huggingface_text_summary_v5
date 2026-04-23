# Hugging Face Text Summary

A small, deployable Streamlit app that:
- Summarizes text using a Hugging Face Transformers model
- Extracts key phrases (YAKE)
- Infers intent via zero-shot classification

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip

# Recommended (installs the package from src/)
pip install -e .

# Optional dev tooling
pip install -r requirements-dev.txt

streamlit run app.py
```

First run will download the models into your Hugging Face cache.

## CLI usage

```powershell
# After `pip install -e .`
hf-text-summary -f input.txt

# from stdin
Get-Content input.txt | hf-text-summary

# (legacy wrapper still works)
python summarize_cli.py -f input.txt
```

## Docker

```powershell
docker build -t hf-text-summary .
docker run --rm -p 8501:8501 hf-text-summary
```

Then open http://localhost:8501

## Notes

- Default summarization model: `sshleifer/distilbart-cnn-12-6` (CPU-friendly)
- Default intent model: `valhalla/distilbart-mnli-12-1`
- For long text, the summarizer uses a chunked (map-reduce) strategy.

If deployment environments are resource-constrained, consider switching to smaller models in the sidebar.

## Python install note (Windows)

If your terminal says "Python was not found", install Python 3.10+ from https://www.python.org/downloads/ and ensure it's on PATH.
