from hf_text_summary.analysis import extract_key_phrases


def test_extract_key_phrases_dedup_and_limit():
    text = "Streamlit app. Streamlit app. Hugging Face models for summarization and intent."
    phrases = extract_key_phrases(text, top_k=5)
    assert len(phrases) <= 5
    assert len({p.lower() for p in phrases}) == len(phrases)
