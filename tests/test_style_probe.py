"""Tests for the adversarial style probe."""

from rhyme_bench.style_probe import (
    StyleFeatureExtractor,
    _incident_to_text,
    _strip_content_tokens,
    run_style_probe,
)


def test_strip_content_tokens():
    text = "Service payment-service had 500 errors at 10:30"
    stripped = _strip_content_tokens(text)
    assert "payment" not in stripped
    assert "W" in stripped
    assert "N" in stripped


def test_strip_preserves_punctuation():
    text = "Error: -- connection failed; timeout"
    stripped = _strip_content_tokens(text)
    assert "P" in stripped  # "--" and ";" are pure punctuation tokens


def test_style_feature_extractor_shape():
    ext = StyleFeatureExtractor()
    texts = ["Hello world. This is a test.", "Another one."]
    features = ext.transform(texts)
    assert features.shape == (2, 17)


def test_style_feature_extractor_values():
    ext = StyleFeatureExtractor()
    features = ext.transform(["Short. Very short."])
    # mean_sent_len should be small
    assert features[0][0] < 5  # mean sentence length


def test_incident_to_text_includes_summary_and_alerts(small_corpus):
    text = _incident_to_text(small_corpus, 0)
    inc = small_corpus.incidents[0]
    assert inc.payload.summary in text
    # Should include alert messages
    assert "[critical]" in text or "[warning]" in text or "[info]" in text


def test_run_style_probe_returns_required_keys(small_corpus):
    results = run_style_probe(small_corpus, cv_folds=3)
    assert "style_accuracy" in results
    assert "char_ngram_accuracy" in results
    assert "numeric_accuracy" in results
    assert "summary_ngram_accuracy" in results
    assert "random_baseline" in results
    assert "threshold" in results
    assert "pass" in results
    assert isinstance(results["pass"], bool)


def test_run_style_probe_baseline_correct(small_corpus):
    results = run_style_probe(small_corpus, cv_folds=3)
    num_classes = results["num_classes"]
    expected_baseline = 1.0 / num_classes
    assert abs(results["random_baseline"] - expected_baseline) < 0.01
