"""Tests for taxonomy integrity."""

from sifter_bench.taxonomy import (
    CAUSE_FAMILY_MAP,
    CONFUSABLE_PAIRS,
    REMEDIATIONS,
    TAXONOMY,
    CauseClass,
    CauseFamily,
    ConfusabilityTier,
)


def test_all_classes_have_family():
    for cc in CauseClass:
        assert cc in CAUSE_FAMILY_MAP, f"{cc} missing from CAUSE_FAMILY_MAP"


def test_all_classes_have_taxonomy_entry():
    for cc in CauseClass:
        assert cc in TAXONOMY, f"{cc} missing from TAXONOMY"
        info = TAXONOMY[cc]
        assert info.description
        assert len(info.distinguishing_signals) >= 3
        assert info.remediation


def test_all_classes_have_remediation():
    for cc in CauseClass:
        assert cc in REMEDIATIONS
        rem = REMEDIATIONS[cc]
        assert rem.canonical
        assert rem.masks_symptom


def test_confusable_pairs_reference_valid_classes():
    for a, b, tier in CONFUSABLE_PAIRS:
        assert a in CauseClass.__members__.values(), f"Invalid class in pair: {a}"
        assert b in CauseClass.__members__.values(), f"Invalid class in pair: {b}"
        assert a != b, f"Self-pair: {a}"
        assert tier in ConfusabilityTier.__members__.values()


def test_every_class_has_at_least_one_confusable_partner():
    paired = set()
    for a, b, _ in CONFUSABLE_PAIRS:
        paired.add(a)
        paired.add(b)
    for cc in CauseClass:
        assert cc in paired, f"{cc} has no confusable partner"


def test_class_count():
    assert len(CauseClass) == 20


def test_family_count():
    assert len(CauseFamily) == 8
