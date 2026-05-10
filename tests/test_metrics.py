"""
Comprehensive unit tests for src/evaluation/metrics.py

Covers:
  - CNEC inline notation parsing (simple, nested, edge cases)
  - Plain text extraction from CNEC
  - BIOES / BIO tag sequence parsing
  - EntityMetrics dataclass arithmetic
  - EvaluationResult aggregation (micro, macro)
  - Evaluator: exact matching, partial matching, multi-sentence
  - Label mapping / normalisation
  - Convenience functions (evaluate_cnec, evaluate_bioes)
  - Edge cases (empty inputs, single tokens, all-O, mismatched types)
"""

# run with `pytest tests/test_metrics.py`


import pytest

from src.evaluation.metrics import (
    CNEC_CLASSES,
    CNEC_SUPERTYPE,
    STUDIO_LABELS,
    Entity,
    EntityMetrics,
    EvaluationResult,
    Evaluator,
    evaluate_bioes,
    evaluate_cnec,
    get_plain_text_cnec,
    parse_bioes_tags,
    parse_cnec_entity,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstants:
    def test_cnec_classes_not_empty(self):
        assert len(CNEC_CLASSES) >= 50

    def test_cnec_classes_contains_known_tags(self):
        for tag in ("pf", "ps", "gc", "gu", "oa", "ic", "P", "T"):
            assert tag in CNEC_CLASSES

    def test_STUDIO_LABELS_not_empty(self):
        assert len(STUDIO_LABELS) == 14

    def test_STUDIO_LABELS_contains_known(self):
        for lbl in ("per", "loc_c", "ins", "tim"):
            assert lbl in STUDIO_LABELS

    def test_cnec_supertype_maps_all_classes(self):
        for tag in CNEC_CLASSES:
            assert tag in CNEC_SUPERTYPE

    def test_cnec_supertype_single_char_maps_to_self(self):
        for tag in ("A", "C", "P", "T"):
            assert CNEC_SUPERTYPE[tag] == tag

    def test_cnec_supertype_two_char_maps_to_upper_first(self):
        assert CNEC_SUPERTYPE["pf"] == "P"
        assert CNEC_SUPERTYPE["gc"] == "G"
        assert CNEC_SUPERTYPE["oa"] == "O"
        assert CNEC_SUPERTYPE["ic"] == "I"


# ═══════════════════════════════════════════════════════════════════════════════
#  CNEC inline parsing
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseCnecEntity:
    def test_no_entities(self):
        assert parse_cnec_entity("Hello world") == []

    def test_single_entity(self):
        ents = parse_cnec_entity("<pf Václav>")
        assert len(ents) == 1
        assert ents[0] == (0, 6, "pf")

    def test_two_entities(self):
        ents = parse_cnec_entity("<pf Jan> a <ps Novák>")
        assert len(ents) == 2
        assert ents[0][2] == "pf"
        assert ents[1][2] == "ps"

    def test_nested_entity(self):
        # <P<pf Jan> <ps Novák>> → P wrapping pf + ps
        ents = parse_cnec_entity("<P<pf Jan> <ps Novák>>")
        labels = {e[2] for e in ents}
        assert "P" in labels
        assert "pf" in labels
        assert "ps" in labels
        assert len(ents) == 3

    def test_nested_entity_spans_correct(self):
        ents = parse_cnec_entity("<P<pf Jan> <ps Novák>>")
        by_label = {e[2]: e for e in ents}
        # Plain text: "Jan Novák"
        assert by_label["pf"] == (0, 3, "pf")
        assert by_label["ps"] == (4, 9, "ps")
        assert by_label["P"] == (0, 9, "P")

    def test_entity_at_start(self):
        ents = parse_cnec_entity("<gc Česko> je stát")
        assert ents[0][2] == "gc"

    def test_entity_at_end(self):
        ents = parse_cnec_entity("bydlí v <gu Praze>")
        assert len(ents) == 1
        assert ents[0][2] == "gu"

    def test_adjacent_entities(self):
        ents = parse_cnec_entity("<pf Jan> <ps Novák>")
        assert len(ents) == 2

    def test_entity_with_multiple_words(self):
        ents = parse_cnec_entity("<gu Karlovy Vary>")
        assert len(ents) == 1
        plain = get_plain_text_cnec("<gu Karlovy Vary>")
        assert plain == "Karlovy Vary"

    def test_deep_nesting(self):
        # <io<s CIA>> → io wrapping s
        ents = parse_cnec_entity("<io<s CIA>>")
        labels = {e[2] for e in ents}
        assert "io" in labels
        # 's' is not in CNEC_CLASSES but is in the special set
        # so it's parsed but not added to entities
        assert len([e for e in ents if e[2] in CNEC_CLASSES]) >= 1

    def test_special_tags_not_in_entities(self):
        # Tags like 'cap', 'lower', 'segm', 's', 'f', '?' should be parsed
        # but only CNEC_CLASSES tags should appear in output entities
        ents = parse_cnec_entity("<cap TO>")
        assert len(ents) == 0  # 'cap' not in CNEC_CLASSES

    def test_literal_angle_bracket_when_not_tag(self):
        # '<' followed by something not a known tag → literal
        ents = parse_cnec_entity("a < b > c")
        assert len(ents) == 0

    def test_empty_string(self):
        assert parse_cnec_entity("") == []

    def test_real_cnec_line_simple(self):
        line = "I s <p_ Dubenkou> , na kterou <if U tygra> teď myslím . . ."
        ents = parse_cnec_entity(line)
        labels = [e[2] for e in ents]
        assert "p_" in labels
        assert "if" in labels

    def test_real_cnec_line_complex(self):
        line = '<P<pf Václavem> <ps Havlem>>'
        ents = parse_cnec_entity(line)
        assert len(ents) == 3
        by_label = {e[2]: e for e in ents}
        assert by_label["P"][0] == 0  # outer starts at 0
        assert by_label["pf"][2] == "pf"
        assert by_label["ps"][2] == "ps"


# ═══════════════════════════════════════════════════════════════════════════════
#  Plain text extraction
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetPlainText:
    def test_no_entities(self):
        assert get_plain_text_cnec("Hello world") == "Hello world"

    def test_single_entity_stripped(self):
        assert get_plain_text_cnec("<pf Jan>") == "Jan"

    def test_nested_stripped(self):
        assert get_plain_text_cnec("<P<pf Jan> <ps Novák>>") == "Jan Novák"

    def test_mixed_text_and_entities(self):
        result = get_plain_text_cnec("bydlí v <gu Praze> .")
        assert result == "bydlí v Praze ."

    def test_multiple_entities(self):
        result = get_plain_text_cnec("<pf Jan> a <pf Marie>")
        assert result == "Jan a Marie"

    def test_empty(self):
        assert get_plain_text_cnec("") == ""

    def test_special_tags_stripped(self):
        assert get_plain_text_cnec("<cap TO>") == "TO"

    def test_preserves_spaces(self):
        result = get_plain_text_cnec("A <pf B> C")
        assert result == "A B C"


# ═══════════════════════════════════════════════════════════════════════════════
#  BIOES parsing
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseBioesTags:
    def test_empty(self):
        assert parse_bioes_tags([]) == []

    def test_all_o(self):
        assert parse_bioes_tags(["O", "O", "O"]) == []

    def test_single_s_tag(self):
        ents = parse_bioes_tags(["S-PER"])
        assert ents == [(0, 1, "PER")]

    def test_be_pair(self):
        ents = parse_bioes_tags(["B-PER", "E-PER"])
        assert ents == [(0, 2, "PER")]

    def test_bie_sequence(self):
        ents = parse_bioes_tags(["B-ORG", "I-ORG", "E-ORG"])
        assert ents == [(0, 3, "ORG")]

    def test_biie_sequence(self):
        ents = parse_bioes_tags(["B-LOC", "I-LOC", "I-LOC", "E-LOC"])
        assert ents == [(0, 4, "LOC")]

    def test_multiple_entities(self):
        tags = ["B-PER", "E-PER", "O", "S-LOC", "O", "B-ORG", "I-ORG", "E-ORG"]
        ents = parse_bioes_tags(tags)
        assert ents == [(0, 2, "PER"), (3, 4, "LOC"), (5, 8, "ORG")]

    def test_adjacent_entities(self):
        tags = ["S-PER", "S-LOC"]
        ents = parse_bioes_tags(tags)
        assert ents == [(0, 1, "PER"), (1, 2, "LOC")]

    def test_bio_format_no_e(self):
        # BIO without E: entity should close at next B or O
        tags = ["B-PER", "I-PER", "O"]
        ents = parse_bioes_tags(tags)
        assert ents == [(0, 2, "PER")]

    def test_bio_b_starts_new(self):
        tags = ["B-PER", "I-PER", "B-LOC", "O"]
        ents = parse_bioes_tags(tags)
        assert len(ents) == 2
        assert ents[0] == (0, 2, "PER")
        assert ents[1] == (2, 3, "LOC")

    def test_bio_unclosed_at_end(self):
        tags = ["B-PER", "I-PER"]
        ents = parse_bioes_tags(tags)
        assert ents == [(0, 2, "PER")]

    def test_recovery_i_without_b(self):
        # I-PER without preceding B-PER → start new entity
        tags = ["I-PER", "I-PER", "O"]
        ents = parse_bioes_tags(tags)
        assert len(ents) == 1
        assert ents[0] == (0, 2, "PER")

    def test_recovery_i_label_mismatch(self):
        tags = ["B-PER", "I-LOC", "O"]
        ents = parse_bioes_tags(tags)
        assert len(ents) == 2
        assert ents[0] == (0, 1, "PER")
        assert ents[1] == (1, 2, "LOC")

    def test_recovery_e_without_b(self):
        tags = ["O", "E-PER", "O"]
        ents = parse_bioes_tags(tags)
        assert len(ents) == 1
        assert ents[0] == (1, 2, "PER")

    def test_malformed_tag_no_dash(self):
        tags = ["B-PER", "MALFORMED", "O"]
        ents = parse_bioes_tags(tags)
        assert len(ents) == 1
        assert ents[0] == (0, 1, "PER")

    def test_lowercase_tags(self):
        tags = ["b-PER", "e-PER"]
        ents = parse_bioes_tags(tags)
        assert ents == [(0, 2, "PER")]

    def test_only_o(self):
        assert parse_bioes_tags(["O"]) == []

    def test_single_b_at_end(self):
        ents = parse_bioes_tags(["O", "B-PER"])
        assert ents == [(1, 2, "PER")]

    def test_hyphenated_label(self):
        # Labels can contain hyphens: "B-loc_c" etc.
        tags = ["B-loc_c", "E-loc_c"]
        ents = parse_bioes_tags(tags)
        assert ents == [(0, 2, "loc_c")]

    def test_s_closes_open_entity(self):
        tags = ["B-PER", "I-PER", "S-LOC"]
        ents = parse_bioes_tags(tags)
        assert len(ents) == 2
        assert ents[0] == (0, 2, "PER")
        assert ents[1] == (2, 3, "LOC")

    def test_long_sequence(self):
        # 100 tokens: alternating entities and O
        tags = []
        expected = []
        for i in range(0, 100, 4):
            tags.extend(["B-PER", "E-PER", "O", "O"])
            expected.append((i, i + 2, "PER"))
        ents = parse_bioes_tags(tags)
        assert ents == expected


# ═══════════════════════════════════════════════════════════════════════════════
#  EntityMetrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestEntityMetrics:
    def test_zero_counts(self):
        m = EntityMetrics()
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.support == 0

    def test_perfect_score(self):
        m = EntityMetrics(tp=10, fp=0, fn=0)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.support == 10

    def test_no_predictions(self):
        m = EntityMetrics(tp=0, fp=0, fn=10)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_all_false_positives(self):
        m = EntityMetrics(tp=0, fp=10, fn=0)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_balanced_errors(self):
        m = EntityMetrics(tp=5, fp=5, fn=5)
        assert m.precision == 0.5
        assert m.recall == 0.5
        assert m.f1 == 0.5

    def test_high_precision_low_recall(self):
        m = EntityMetrics(tp=1, fp=0, fn=9)
        assert m.precision == 1.0
        assert m.recall == 0.1
        assert abs(m.f1 - 2 / 11) < 1e-9

    def test_support_is_tp_plus_fn(self):
        m = EntityMetrics(tp=3, fp=7, fn=2)
        assert m.support == 5


# ═══════════════════════════════════════════════════════════════════════════════
#  EvaluationResult
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluationResult:
    def test_micro_from_exact_match(self):
        r = EvaluationResult(
            exact_match=EntityMetrics(tp=8, fp=2, fn=2),
        )
        assert r.micro_precision == 0.8
        assert r.micro_recall == 0.8
        assert abs(r.micro_f1 - 0.8) < 1e-9

    def test_macro_averages_types(self):
        r = EvaluationResult(
            per_type={
                "PER": EntityMetrics(tp=10, fp=0, fn=0),  # F1 = 1.0
                "LOC": EntityMetrics(tp=0, fp=0, fn=10),  # F1 = 0.0
            },
        )
        assert r.macro_f1 == 0.5
        assert r.macro_precision == 0.5
        assert r.macro_recall == 0.5

    def test_macro_skips_zero_support(self):
        r = EvaluationResult(
            per_type={
                "PER": EntityMetrics(tp=10, fp=0, fn=0),
                "LOC": EntityMetrics(tp=0, fp=5, fn=0),  # support=0
            },
        )
        # Only PER has support > 0, so macro = PER's metrics
        assert r.macro_f1 == 1.0

    def test_macro_empty(self):
        r = EvaluationResult()
        assert r.macro_f1 == 0.0

    def test_summary_returns_string(self):
        r = EvaluationResult(
            per_type={"PER": EntityMetrics(tp=5, fp=1, fn=1)},
            exact_match=EntityMetrics(tp=5, fp=1, fn=1),
            partial_match=EntityMetrics(tp=6, fp=0, fn=1),
        )
        s = r.summary()
        assert isinstance(s, str)
        assert "PER" in s
        assert "micro" in s
        assert "macro" in s
        assert "partial" in s

    def test_summary_contains_all_types(self):
        r = EvaluationResult(
            per_type={
                "PER": EntityMetrics(tp=5, fp=0, fn=0),
                "LOC": EntityMetrics(tp=3, fp=1, fn=2),
                "ORG": EntityMetrics(tp=0, fp=0, fn=1),
            },
            exact_match=EntityMetrics(tp=8, fp=1, fn=3),
        )
        s = r.summary()
        assert "PER" in s
        assert "LOC" in s
        assert "ORG" in s


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluator — Exact matching
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluatorExact:
    def test_perfect_match(self):
        ev = Evaluator()
        gold = [(0, 3, "PER"), (5, 8, "LOC")]
        ev.add(gold, gold)
        r = ev.result()
        assert r.exact_match.tp == 2
        assert r.exact_match.fp == 0
        assert r.exact_match.fn == 0
        assert r.micro_f1 == 1.0

    def test_no_predictions(self):
        ev = Evaluator()
        ev.add([(0, 3, "PER")], [])
        r = ev.result()
        assert r.exact_match.tp == 0
        assert r.exact_match.fn == 1
        assert r.exact_match.fp == 0

    def test_no_gold(self):
        ev = Evaluator()
        ev.add([], [(0, 3, "PER")])
        r = ev.result()
        assert r.exact_match.tp == 0
        assert r.exact_match.fp == 1
        assert r.exact_match.fn == 0

    def test_both_empty(self):
        ev = Evaluator()
        ev.add([], [])
        r = ev.result()
        assert r.exact_match.tp == 0
        assert r.exact_match.fp == 0
        assert r.exact_match.fn == 0

    def test_wrong_type(self):
        ev = Evaluator()
        ev.add([(0, 3, "PER")], [(0, 3, "LOC")])
        r = ev.result()
        assert r.exact_match.tp == 0
        assert r.exact_match.fp == 1
        assert r.exact_match.fn == 1

    def test_wrong_span_start(self):
        ev = Evaluator()
        ev.add([(0, 3, "PER")], [(1, 3, "PER")])
        r = ev.result()
        assert r.exact_match.tp == 0

    def test_wrong_span_end(self):
        ev = Evaluator()
        ev.add([(0, 3, "PER")], [(0, 4, "PER")])
        r = ev.result()
        assert r.exact_match.tp == 0

    def test_partial_overlap_not_exact(self):
        ev = Evaluator()
        ev.add([(0, 5, "PER")], [(2, 7, "PER")])
        r = ev.result()
        assert r.exact_match.tp == 0
        assert r.exact_match.fp == 1
        assert r.exact_match.fn == 1

    def test_multiple_sentences(self):
        ev = Evaluator()
        ev.add([(0, 2, "PER")], [(0, 2, "PER")])  # 1 TP
        ev.add([(0, 3, "LOC")], [])                # 1 FN
        ev.add([], [(0, 1, "ORG")])                 # 1 FP
        r = ev.result()
        assert r.exact_match.tp == 1
        assert r.exact_match.fn == 1
        assert r.exact_match.fp == 1

    def test_per_type_tracking(self):
        ev = Evaluator()
        ev.add(
            [(0, 2, "PER"), (3, 5, "LOC"), (6, 8, "ORG")],
            [(0, 2, "PER"), (3, 5, "LOC")],
        )
        r = ev.result()
        assert r.per_type["PER"].tp == 1
        assert r.per_type["LOC"].tp == 1
        assert r.per_type["ORG"].fn == 1
        assert r.per_type["ORG"].tp == 0

    def test_duplicate_entities_in_pred(self):
        ev = Evaluator()
        ev.add([(0, 2, "PER")], [(0, 2, "PER"), (0, 2, "PER")])
        r = ev.result()
        assert r.exact_match.tp == 1
        assert r.exact_match.fp == 1  # second pred is FP

    def test_duplicate_entities_in_gold(self):
        ev = Evaluator()
        ev.add([(0, 2, "PER"), (0, 2, "PER")], [(0, 2, "PER")])
        r = ev.result()
        assert r.exact_match.tp == 1
        assert r.exact_match.fn == 1  # second gold is FN


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluator — Partial matching
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluatorPartial:
    def test_exact_is_also_partial(self):
        ev = Evaluator()
        ev.add([(0, 5, "PER")], [(0, 5, "PER")])
        r = ev.result()
        assert r.partial_match.tp == 1

    def test_overlapping_same_type_is_partial(self):
        ev = Evaluator()
        ev.add([(0, 5, "PER")], [(2, 7, "PER")])
        r = ev.result()
        assert r.partial_match.tp == 1
        assert r.exact_match.tp == 0

    def test_overlapping_diff_type_not_partial(self):
        ev = Evaluator()
        ev.add([(0, 5, "PER")], [(2, 7, "LOC")])
        r = ev.result()
        assert r.partial_match.tp == 0

    def test_non_overlapping_not_partial(self):
        ev = Evaluator()
        ev.add([(0, 3, "PER")], [(5, 8, "PER")])
        r = ev.result()
        assert r.partial_match.tp == 0

    def test_adjacent_not_partial(self):
        # [0,3) and [3,6) don't overlap
        ev = Evaluator()
        ev.add([(0, 3, "PER")], [(3, 6, "PER")])
        r = ev.result()
        assert r.partial_match.tp == 0

    def test_contained_span_is_partial(self):
        ev = Evaluator()
        ev.add([(0, 10, "PER")], [(2, 5, "PER")])
        r = ev.result()
        assert r.partial_match.tp == 1


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluator — Label mapping
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluatorLabelMap:
    def test_label_map_normalises(self):
        ev = Evaluator(label_map={"pf": "PER", "ps": "PER"})
        ev.add([(0, 3, "pf")], [(0, 3, "ps")])
        r = ev.result()
        # Both map to "PER", so it's a TP
        assert r.exact_match.tp == 1

    def test_label_map_unmapped_stays(self):
        ev = Evaluator(label_map={"pf": "PER"})
        ev.add([(0, 3, "LOC")], [(0, 3, "LOC")])
        r = ev.result()
        assert r.exact_match.tp == 1

    def test_label_map_mismatch_after_mapping(self):
        ev = Evaluator(label_map={"pf": "PER", "gc": "LOC"})
        ev.add([(0, 3, "pf")], [(0, 3, "gc")])
        r = ev.result()
        assert r.exact_match.tp == 0  # PER != LOC

    def test_cnec_supertype_map(self):
        ev = Evaluator(label_map=CNEC_SUPERTYPE)
        ev.add([(0, 3, "pf")], [(0, 3, "ps")])
        r = ev.result()
        # Both map to "P"
        assert r.exact_match.tp == 1


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluator — Reset
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluatorReset:
    def test_reset_clears(self):
        ev = Evaluator()
        ev.add([(0, 3, "PER")], [(0, 3, "PER")])
        ev.reset()
        r = ev.result()
        assert r.exact_match.tp == 0
        assert r.exact_match.fp == 0
        assert r.exact_match.fn == 0
        assert len(r.per_type) == 0

    def test_reset_then_add(self):
        ev = Evaluator()
        ev.add([(0, 3, "PER")], [])
        ev.reset()
        ev.add([(0, 3, "LOC")], [(0, 3, "LOC")])
        r = ev.result()
        assert r.exact_match.tp == 1
        assert "LOC" in r.per_type
        assert "PER" not in r.per_type


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluator — Macro vs Micro
# ═══════════════════════════════════════════════════════════════════════════════


class TestMacroVsMicro:
    def test_imbalanced_favours_micro(self):
        """When frequent type is perfect but rare type is zero,
        micro > macro because micro is dominated by the frequent type."""
        ev = Evaluator()
        for i in range(100):
            ev.add([(i, i + 1, "PER")], [(i, i + 1, "PER")])
        for i in range(10):
            ev.add([(i, i + 1, "ORG")], [])  # all missed
        r = ev.result()
        assert r.micro_f1 > r.macro_f1

    def test_balanced_types_equal(self):
        """When both types have equal performance, micro ≈ macro."""
        ev = Evaluator()
        for i in range(10):
            ev.add([(i, i + 1, "PER")], [(i, i + 1, "PER")])
            ev.add([(i, i + 1, "LOC")], [(i, i + 1, "LOC")])
        r = ev.result()
        assert abs(r.micro_f1 - r.macro_f1) < 0.01

    def test_macro_one_perfect_one_zero(self):
        ev = Evaluator()
        for i in range(10):
            ev.add(
                [(i * 10, i * 10 + 3, "PER"), (i * 10 + 5, i * 10 + 8, "LOC")],
                [(i * 10, i * 10 + 3, "PER")],
            )
        r = ev.result()
        assert abs(r.macro_f1 - 0.5) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience: evaluate_cnec
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluateCnecInline:
    def test_perfect_self_comparison(self):
        lines = ["<pf Jan> bydlí v <gu Praze>", "<ps Novák> pracuje"]
        r = evaluate_cnec(lines, lines)
        assert r.micro_f1 == 1.0

    def test_empty_pred(self):
        gold = ["<pf Jan>"]
        pred = ["Jan"]
        r = evaluate_cnec(gold, pred)
        assert r.micro_f1 == 0.0
        assert r.exact_match.fn == 1

    def test_line_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="Line count mismatch"):
            evaluate_cnec(["<pf Jan>"], ["Jan", "extra"])

    def test_with_label_map(self):
        gold = ["<pf Jan>"]
        pred = ["<ps Jan>"]
        r = evaluate_cnec(gold, pred, label_map={"pf": "P", "ps": "P"})
        assert r.micro_f1 == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience: evaluate_bioes
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluateBioes:
    def test_perfect(self):
        gold = [["B-PER", "E-PER", "O"]]
        pred = [["B-PER", "E-PER", "O"]]
        r = evaluate_bioes(gold, pred)
        assert r.micro_f1 == 1.0

    def test_all_missed(self):
        gold = [["B-PER", "E-PER", "O"]]
        pred = [["O", "O", "O"]]
        r = evaluate_bioes(gold, pred)
        assert r.micro_f1 == 0.0

    def test_sequence_count_mismatch(self):
        with pytest.raises(ValueError, match="Sequence count mismatch"):
            evaluate_bioes([["O"]], [["O"], ["O"]])

    def test_tag_count_mismatch(self):
        with pytest.raises(ValueError, match="Tag count mismatch"):
            evaluate_bioes([["O", "O"]], [["O"]])

    def test_multiple_sentences(self):
        gold = [["S-PER", "O"], ["O", "S-LOC"]]
        pred = [["S-PER", "O"], ["O", "S-LOC"]]
        r = evaluate_bioes(gold, pred)
        assert r.exact_match.tp == 2

    def test_with_label_map(self):
        gold = [["S-pf", "O"]]
        pred = [["S-ps", "O"]]
        r = evaluate_bioes(gold, pred, label_map={"pf": "P", "ps": "P"})
        assert r.micro_f1 == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Integration: BIOES → Evaluator round-trip
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_bioes_to_evaluator(self):
        """Parse BIOES tags, then evaluate via Evaluator."""
        gold_tags = ["B-PER", "E-PER", "O", "S-LOC", "O"]
        pred_tags = ["B-PER", "E-PER", "O", "O", "O"]

        gold_ents = parse_bioes_tags(gold_tags)
        pred_ents = parse_bioes_tags(pred_tags)

        ev = Evaluator()
        ev.add(gold_ents, pred_ents)
        r = ev.result()

        assert r.exact_match.tp == 1  # PER matched
        assert r.exact_match.fn == 1  # LOC missed
        assert r.exact_match.fp == 0
        assert r.per_type["PER"].f1 == 1.0
        assert r.per_type["LOC"].f1 == 0.0

    def test_cnec_to_evaluator(self):
        """Parse CNEC inline, then evaluate."""
        gold = "<pf Jan> pracuje v <gu Brně>"
        pred = "<pf Jan> pracuje v Brně"

        gold_ents = parse_cnec_entity(gold)
        pred_ents = parse_cnec_entity(pred)

        ev = Evaluator()
        ev.add(gold_ents, pred_ents)
        r = ev.result()

        assert r.exact_match.tp == 1  # pf matched
        assert r.exact_match.fn == 1  # gu missed

    def test_many_sentences_accumulation(self):
        ev = Evaluator()
        for i in range(50):
            ev.add([(i, i + 1, "PER")], [(i, i + 1, "PER")])
        for i in range(50):
            ev.add([(i, i + 1, "LOC")], [])
        r = ev.result()
        assert r.exact_match.tp == 50
        assert r.exact_match.fn == 50
        assert r.exact_match.fp == 0
        # PER: P=1.0, R=1.0, F1=1.0; LOC: F1=0.0
        assert abs(r.macro_f1 - 0.5) < 0.01
        # Micro: P=1.0, R=0.5, F1=0.667
        assert abs(r.micro_f1 - 2 / 3) < 0.01

    def test_summary_output_format(self):
        ev = Evaluator()
        ev.add(
            [(0, 2, "PER"), (3, 5, "LOC")],
            [(0, 2, "PER"), (3, 6, "LOC")],
        )
        r = ev.result()
        s = r.summary()
        lines = s.strip().split("\n")
        assert len(lines) >= 6  # header + separator + types + separator + micro + macro + partial
        assert "PER" in s
        assert "LOC" in s
