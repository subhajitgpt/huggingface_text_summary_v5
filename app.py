from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import streamlit as st


def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


# If the user runs `python app.py`, Streamlit session runtime won't exist and
# they'll see warnings like: "Session state does not function when running a script
# without `streamlit run`". Relaunch seamlessly.
if __name__ == "__main__" and not _running_in_streamlit():
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
        )
    )

try:
    from hf_text_summary import analyze_text, DEFAULT_INTENT_MODEL, DEFAULT_SUMMARY_MODEL
except ModuleNotFoundError:
    # Allows `streamlit run app.py` without requiring an editable install.
    import sys

    sys.path.append(str(Path(__file__).parent / "src"))
    from hf_text_summary import analyze_text, DEFAULT_INTENT_MODEL, DEFAULT_SUMMARY_MODEL


DEFAULT_INTENT_LABELS = [
    "requesting information",
    "task request",
    "complaint",
    "feedback",
    "bug report",
    "purchase inquiry",
    "meeting scheduling",
    "status update",
    "follow-up",
    "other",
]

SAMPLE_TEXT = """\
Our mobile app started crashing after yesterday's update. Users report that it closes immediately when opening the Settings screen.\

This seems to happen on Android 14 devices, especially on Pixel phones. We need a quick fix or a rollback plan.\

Can you investigate the logs, identify the root cause, and propose a patch along with an ETA?\
"""


st.set_page_config(page_title="Hugging Face Text Summary", layout="wide")

st.title("Hugging Face Text Summary")
st.caption("Summarize text, extract key phrases, and infer intent (zero-shot).")

with st.sidebar:
    st.subheader("Models")
    summary_model = st.text_input(
        "Summarization model",
        value=DEFAULT_SUMMARY_MODEL,
        help="Any Hugging Face summarization model compatible with Transformers pipelines.",
    )
    intent_model = st.text_input(
        "Intent model (zero-shot)",
        value=DEFAULT_INTENT_MODEL,
        help="A NLI/zero-shot model. Keep it small for CPU deploys.",
    )

    st.subheader("Runtime")
    device = st.selectbox(
        "Device",
        options=["cpu", "cuda"],
        index=0,
        help="Use 'cuda' only if a GPU + CUDA Torch is available.",
    )

    st.subheader("Summary")
    summary_min = st.slider("Min length", 10, 200, 40, step=5)
    summary_max = st.slider("Max length", 30, 400, 160, step=10)
    summary_refine_final = st.checkbox(
        "Refine final summary (slower)",
        value=True,
        help="For long text, an extra pass improves coherence but costs time.",
    )

    st.subheader("Key phrases")
    keyphrase_top_k = st.slider("Count", 3, 20, 10)

    st.subheader("Intent")
    enable_intent = st.checkbox(
        "Enable intent detection (slower)",
        value=True,
        help="Zero-shot intent is usually the slowest step on CPU.",
    )
    intent_labels_text = st.text_area(
        "Candidate intents (one per line)",
        value="\n".join(DEFAULT_INTENT_LABELS),
        height=180,
    )
    intent_top_k = st.slider("Top-K intents", 1, 5, 3)

    run = st.button("Analyze", type="primary", use_container_width=True)

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.subheader("Input")
    text = st.text_area(
        "Text",
        value=SAMPLE_TEXT,
        height=260,
        placeholder="Paste text to summarize...",
        label_visibility="collapsed",
    )

with col_right:
    st.subheader("Tips")
    st.write(
        "- For long inputs, the app uses a chunked (map-reduce) summary.\n"
        "- Change intent labels to fit your domain.\n"
        "- First run downloads models into your Hugging Face cache."
    )

if run:
    cleaned = (text or "").strip()
    labels: List[str] = [
        line.strip() for line in (intent_labels_text or "").splitlines() if line.strip()
    ]

    if not cleaned:
        st.warning("Please enter some text.")
        st.stop()

    if summary_max <= summary_min:
        st.warning("Summary max length must be greater than min length.")
        st.stop()

    with st.spinner("Running models..."):
        result = analyze_text(
            cleaned,
            summary_model=summary_model,
            intent_model=intent_model,
            device=device,
            summary_min_length=summary_min,
            summary_max_length=summary_max,
            summary_refine_final=summary_refine_final,
            keyphrase_top_k=keyphrase_top_k,
            enable_intent=enable_intent,
            intent_labels=labels,
            intent_top_k=intent_top_k,
        )

    st.divider()

    out_left, out_right = st.columns([3, 2], gap="large")

    with out_left:
        st.subheader("Summary")
        if result.summary:
            st.write(result.summary)
        else:
            st.info("No summary produced (input may be too short).")

    with out_right:
        st.subheader("Key phrases")
        if result.key_phrases:
            for p in result.key_phrases:
                st.markdown(f"- {p}")
        else:
            st.info("No key phrases extracted.")

        st.subheader("Intent")
        if result.intent_top:
            st.markdown(
                f"**Top:** {result.intent_top.label}  \\nScore: `{result.intent_top.score:.3f}`"
            )
            if result.intent_top_k:
                st.caption("Top-K")
                for pred in result.intent_top_k:
                    st.markdown(f"- {pred.label}: `{pred.score:.3f}`")
        else:
            st.info("No intent prediction (check intent labels).")

    with st.expander("Metadata", expanded=False):
        st.json(result.meta)

else:
    st.info("Click **Analyze** to generate a summary.")
