from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any

import re

import yake
from transformers import pipeline


DEFAULT_SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"
# Smaller + typically faster on CPU than many BART-based MNLI models.
DEFAULT_INTENT_MODEL = "typeform/distilbert-base-uncased-mnli"


@dataclass(frozen=True)
class IntentPrediction:
    label: str
    score: float


@dataclass(frozen=True)
class AnalysisResult:
    summary: str
    key_phrases: List[str]
    intent_top: Optional[IntentPrediction]
    intent_top_k: List[IntentPrediction]
    meta: Dict[str, Any]


def _clean_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _device_to_pipeline_arg(device: str) -> int:
    device = (device or "cpu").strip().lower()
    if device in {"cpu", "-1"}:
        return -1
    if device in {"cuda", "gpu", "0"}:
        return 0
    raise ValueError("device must be 'cpu' or 'cuda'")


@lru_cache(maxsize=4)
def _summarization_pipeline(model_name: str, device: str):
    return pipeline(
        task="summarization",
        model=model_name,
        device=_device_to_pipeline_arg(device),
    )


@lru_cache(maxsize=4)
def _intent_pipeline(model_name: str, device: str):
    return pipeline(
        task="zero-shot-classification",
        model=model_name,
        device=_device_to_pipeline_arg(device),
    )


def _iter_paragraphs(text: str) -> Iterable[str]:
    for part in re.split(r"\n\s*\n", text):
        part = part.strip()
        if part:
            yield part


def _chunk_by_tokens(text: str, tokenizer, max_input_tokens: int) -> List[str]:
    paragraphs = list(_iter_paragraphs(text))
    if not paragraphs:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def token_len(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    for para in paragraphs:
        para_tokens = token_len(para)

        if para_tokens > max_input_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                sent_tokens = token_len(sent)
                if sent_tokens > max_input_tokens:
                    for i in range(0, len(sent), 800):
                        slice_ = sent[i : i + 800].strip()
                        if slice_:
                            chunks.append(slice_)
                    continue

                if current and current_tokens + sent_tokens > max_input_tokens:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_tokens = 0

                current.append(sent)
                current_tokens += sent_tokens
            continue

        if current and current_tokens + para_tokens > max_input_tokens:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _summarize_one(
    summarizer,
    text: str,
    min_length: int,
    max_length: int,
) -> str:
    out = summarizer(
        text,
        min_length=min_length,
        max_length=max_length,
        do_sample=False,
        truncation=True,
    )
    if not out:
        return ""
    return (out[0].get("summary_text") or "").strip()


def summarize_text(
    text: str,
    *,
    model_name: str = DEFAULT_SUMMARY_MODEL,
    device: str = "cpu",
    max_input_tokens: int = 900,
    min_length: int = 40,
    max_length: int = 160,
    refine_final_summary: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Summarize text using a chunked (map-reduce) approach.

    Returns (summary, meta).
    """

    text = _clean_text(text)
    if not text:
        return "", {"chunks": 0, "model": model_name}

    summarizer = _summarization_pipeline(model_name, device)
    tokenizer = summarizer.tokenizer

    chunks = _chunk_by_tokens(text, tokenizer, max_input_tokens=max_input_tokens)
    if not chunks:
        return "", {"chunks": 0, "model": model_name}

    chunk_summaries = [
        _summarize_one(summarizer, chunk, min_length=min_length, max_length=max_length)
        for chunk in chunks
    ]
    chunk_summaries = [s for s in chunk_summaries if s]

    if not chunk_summaries:
        return "", {"chunks": len(chunks), "model": model_name}

    if len(chunk_summaries) == 1:
        return chunk_summaries[0], {"chunks": len(chunks), "model": model_name}

    if not refine_final_summary:
        # Fast-path: avoid the extra summarization pass.
        joined = "\n".join(chunk_summaries)
        return joined, {
            "chunks": len(chunks),
            "model": model_name,
            "stage": "map-only",
        }

    combined = "\n".join(chunk_summaries)
    final = _summarize_one(
        summarizer,
        combined,
        min_length=max(20, min_length // 2),
        max_length=max(60, max_length),
    )

    return final or "\n".join(chunk_summaries), {
        "chunks": len(chunks),
        "model": model_name,
        "stage": "map-reduce",
    }


def extract_key_phrases(
    text: str,
    *,
    top_k: int = 10,
    language: str = "en",
    max_ngram_size: int = 3,
) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []

    kw_extractor = yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        top=top_k,
        dedupLim=0.9,
        dedupFunc="seqm",
    )
    keywords = kw_extractor.extract_keywords(text)

    phrases = [phrase.strip() for phrase, _score in keywords if phrase.strip()]

    seen = set()
    out: List[str] = []
    for p in phrases:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def detect_intent(
    text: str,
    *,
    candidate_labels: Sequence[str],
    model_name: str = DEFAULT_INTENT_MODEL,
    device: str = "cpu",
    top_k: int = 3,
) -> Tuple[Optional[IntentPrediction], List[IntentPrediction], Dict[str, Any]]:
    text = _clean_text(text)
    labels = [l.strip() for l in candidate_labels if l and l.strip()]
    if not text or not labels:
        return None, [], {"model": model_name}

    classifier = _intent_pipeline(model_name, device)
    res = classifier(text, labels, multi_label=False)

    ranked = list(zip(res.get("labels", []), res.get("scores", [])))
    preds = [IntentPrediction(label=l, score=float(s)) for l, s in ranked[: max(1, top_k)]]
    top = preds[0] if preds else None
    return top, preds, {"model": model_name}


def analyze_text(
    text: str,
    *,
    summary_model: str = DEFAULT_SUMMARY_MODEL,
    intent_model: str = DEFAULT_INTENT_MODEL,
    device: str = "cpu",
    summary_min_length: int = 40,
    summary_max_length: int = 160,
    summary_refine_final: bool = True,
    keyphrase_top_k: int = 10,
    enable_intent: bool = True,
    intent_labels: Optional[Sequence[str]] = None,
    intent_top_k: int = 3,
) -> AnalysisResult:
    text = _clean_text(text)

    summary, sum_meta = summarize_text(
        text,
        model_name=summary_model,
        device=device,
        min_length=summary_min_length,
        max_length=summary_max_length,
        refine_final_summary=summary_refine_final,
    )

    key_phrases = extract_key_phrases(text, top_k=keyphrase_top_k)

    if enable_intent:
        labels = intent_labels or (
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
        )

        intent_top, intent_topk, intent_meta = detect_intent(
            text,
            candidate_labels=labels,
            model_name=intent_model,
            device=device,
            top_k=intent_top_k,
        )
    else:
        intent_top, intent_topk, intent_meta = None, [], {"model": intent_model, "skipped": True}

    return AnalysisResult(
        summary=summary,
        key_phrases=key_phrases,
        intent_top=intent_top,
        intent_top_k=intent_topk,
        meta={
            "summary": sum_meta,
            "intent": intent_meta,
            "device": device,
            "chars": len(text),
        },
    )
