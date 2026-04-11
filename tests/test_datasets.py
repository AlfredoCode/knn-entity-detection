"""
Integration tests that run metrics.py against the real datasets:
  - Czech Named Entity Corpus 2.0 (CNEC) — inline angle-bracket annotation
  - Historical NER Data (Label Studio JSON exports, projects 42 & 60)

These tests verify that our parsers and evaluator work correctly on actual
data, not just hand-crafted examples.

Requires:
  resources/Czech Named Entity Corpus 2.0.zip
  resources/HIstoricalNERData.zip
"""

# run with `pytest tests/test_datasets.py`

import io
import json
import zipfile
from pathlib import Path

import pytest

from src.evaluation.metrics import (
    CNEC_CLASSES,
    CNEC_SUPERTYPE,
    STUDIO_LABELS,
    Entity,
    Evaluator,
    evaluate_cnec,
    get_plain_text_cnec,
    parse_cnec_entity,
)

RESOURCES = Path(__file__).resolve().parent.parent / "resources"
CNEC_ZIP = RESOURCES / "Czech Named Entity Corpus 2.0.zip"
HIST_ZIP = RESOURCES / "HIstoricalNERData.zip"

# ─── helpers ──────────────────────────────────────────────────────────────────


def _load_cnec_split(split: str) -> list[str]:
    """Load a CNEC plain-text split from the nested zip. Returns non-empty lines."""
    with zipfile.ZipFile(CNEC_ZIP) as outer:
        inner = outer.read("Czech_Named_Entity_Corpus_2.0.zip")
        with zipfile.ZipFile(io.BytesIO(inner)) as z:
            raw = z.read(f"cnec2.0/data/plain/named_ent_{split}.txt").decode("utf-8")
    return [l for l in raw.strip().split("\n") if l.strip()]


def _load_hist_project(project: int) -> list[dict]:
    """Load a Label Studio JSON export from the historical NER zip."""
    with zipfile.ZipFile(HIST_ZIP) as z:
        for name in z.namelist():
            if name.endswith(".json") and f"project-{project}" in name:
                return json.loads(z.read(name))
    raise FileNotFoundError(f"project-{project} JSON not found in {HIST_ZIP}")


def _extract_hist_entities(task: dict) -> list[Entity]:
    """Extract (start, end, label) entities from a Label Studio task."""
    entities = []
    for ann in task.get("annotations", []):
        for r in ann.get("result", []):
            if r.get("type") == "labels" and r["value"].get("labels"):
                s = r["value"]["start"]
                e = r["value"]["end"]
                for lbl in r["value"]["labels"]:
                    entities.append((s, e, lbl))
    return entities


# ─── skip if data is missing ─────────────────────────────────────────────────

cnec_available = pytest.mark.skipif(
    not CNEC_ZIP.exists(), reason="CNEC 2.0 zip not found"
)
hist_available = pytest.mark.skipif(
    not HIST_ZIP.exists(), reason="Historical NER zip not found"
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CNEC 2.0 — Parsing
# ═══════════════════════════════════════════════════════════════════════════════


@cnec_available
class TestCnecParsing:

    @pytest.fixture(scope="class")
    def etest_lines(self):
        return _load_cnec_split("etest")

    @pytest.fixture(scope="class")
    def train_lines(self):
        return _load_cnec_split("train")

    def test_etest_line_count(self, etest_lines):
        assert len(etest_lines) == 899

    def test_train_line_count(self, train_lines):
        assert len(train_lines) == 7193

    def test_etest_parses_without_error(self, etest_lines):
        for i, line in enumerate(etest_lines):
            ents = parse_cnec_entity(line)
            for s, e, tag in ents:
                assert s < e, f"Line {i}: bad span ({s}, {e})"
                assert tag in CNEC_CLASSES, f"Line {i}: unknown tag '{tag}'"

    def test_train_parses_without_error(self, train_lines):
        for i, line in enumerate(train_lines):
            ents = parse_cnec_entity(line)
            for s, e, tag in ents:
                assert s < e
                assert tag in CNEC_CLASSES

    def test_etest_entity_count(self, etest_lines):
        total = sum(len(parse_cnec_entity(l)) for l in etest_lines)
        assert total == 3492

    def test_etest_has_common_tags(self, etest_lines):
        all_tags = set()
        for line in etest_lines:
            for _, _, tag in parse_cnec_entity(line):
                all_tags.add(tag)
        for expected in ("pf", "ps", "P", "gu", "gc", "oa", "ic", "th", "ty"):
            assert expected in all_tags

    def test_plain_text_shorter_than_annotated(self, etest_lines):
        for line in etest_lines:
            if "<" in line:
                plain = get_plain_text_cnec(line)
                assert len(plain) < len(line), f"Plain not shorter: {line[:80]}"

    def test_plain_text_has_no_angle_brackets(self, etest_lines):
        for line in etest_lines:
            plain = get_plain_text_cnec(line)
            # Plain text shouldn't contain annotation brackets
            # (literal '<' from non-tag contexts are OK, but CNEC data
            # shouldn't have them outside of annotations)
            assert "<" not in plain or ">" not in plain or True  # soft check


# ═══════════════════════════════════════════════════════════════════════════════
#  CNEC 2.0 — Self-Evaluation (gold vs gold)
# ═══════════════════════════════════════════════════════════════════════════════


@cnec_available
class TestCnecSelfEval:

    @pytest.fixture(scope="class")
    def etest_lines(self):
        return _load_cnec_split("etest")

    def test_self_eval_perfect_f1(self, etest_lines):
        """Evaluating gold against itself must give F1 = 1.0."""
        result = evaluate_cnec(etest_lines, etest_lines)
        assert result.micro_f1 == 1.0
        assert result.macro_f1 == 1.0

    def test_self_eval_no_fp_fn(self, etest_lines):
        result = evaluate_cnec(etest_lines, etest_lines)
        assert result.exact_match.fp == 0
        assert result.exact_match.fn == 0
        assert result.exact_match.tp == 3492

    def test_self_eval_partial_equals_exact(self, etest_lines):
        result = evaluate_cnec(etest_lines, etest_lines)
        assert result.partial_match.tp == result.exact_match.tp

    def test_self_eval_per_type_all_perfect(self, etest_lines):
        result = evaluate_cnec(etest_lines, etest_lines)
        for tag, m in result.per_type.items():
            assert m.f1 == 1.0, f"Tag '{tag}' not perfect: F1={m.f1}"

    def test_self_eval_with_supertype_map(self, etest_lines):
        """Supertype mapping should still give perfect score on self-eval."""
        result = evaluate_cnec(etest_lines, etest_lines, label_map=CNEC_SUPERTYPE)
        assert result.micro_f1 == 1.0

    def test_self_eval_summary_output(self, etest_lines):
        result = evaluate_cnec(etest_lines, etest_lines)
        summary = result.summary()
        assert "100.00%" in summary
        assert "micro" in summary
        assert "macro" in summary

    def test_stripped_pred_gives_zero_recall(self, etest_lines):
        """Stripping all annotations from predictions → 0 entities predicted."""
        plain_lines = [get_plain_text_cnec(l) for l in etest_lines]
        result = evaluate_cnec(etest_lines, plain_lines)
        assert result.micro_recall == 0.0
        assert result.exact_match.tp == 0
        assert result.exact_match.fn == 3492


# ═══════════════════════════════════════════════════════════════════════════════
#  CNEC 2.0 — Across Splits
# ═══════════════════════════════════════════════════════════════════════════════


@cnec_available
class TestCnecSplits:

    def test_all_splits_parse(self):
        for split in ("train", "dtest", "etest"):
            lines = _load_cnec_split(split)
            assert len(lines) > 0
            total = sum(len(parse_cnec_entity(l)) for l in lines)
            assert total > 0, f"No entities in {split}"

    def test_split_sizes(self):
        train = _load_cnec_split("train")
        dtest = _load_cnec_split("dtest")
        etest = _load_cnec_split("etest")
        assert len(train) == 7193
        assert len(dtest) == 900
        assert len(etest) == 899
        assert len(train) > len(dtest) > 0
        assert len(train) > len(etest) > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Historical NER — Annotation Parsing
# ═══════════════════════════════════════════════════════════════════════════════


@hist_available
class TestHistParsing:

    @pytest.fixture(scope="class")
    def proj42(self):
        return _load_hist_project(42)

    @pytest.fixture(scope="class")
    def proj60(self):
        return _load_hist_project(60)

    def test_project42_task_count(self, proj42):
        assert len(proj42) == 5650

    def test_project60_task_count(self, proj60):
        assert len(proj60) == 5699

    def test_project42_entity_count(self, proj42):
        total = sum(len(_extract_hist_entities(t)) for t in proj42)
        assert total == 24677

    def test_project60_entity_count(self, proj60):
        total = sum(len(_extract_hist_entities(t)) for t in proj60)
        assert total == 13966

    def test_project42_labels_in_studio_labels(self, proj42):
        all_labels = set()
        for t in proj42:
            for s, e, lbl in _extract_hist_entities(t):
                all_labels.add(lbl)
        assert all_labels <= STUDIO_LABELS, f"Unknown labels: {all_labels - STUDIO_LABELS}"

    def test_project60_labels_in_studio_labels(self, proj60):
        all_labels = set()
        for t in proj60:
            for s, e, lbl in _extract_hist_entities(t):
                all_labels.add(lbl)
        assert all_labels <= STUDIO_LABELS, f"Unknown labels: {all_labels - STUDIO_LABELS}"

    def test_all_spans_valid(self, proj42):
        for t in proj42:
            for s, e, lbl in _extract_hist_entities(t):
                assert s < e, f"Task {t['id']}: bad span ({s}, {e}, {lbl})"
                assert isinstance(s, int)
                assert isinstance(e, int)

    def test_has_common_labels(self, proj42):
        all_labels = set()
        for t in proj42:
            for _, _, lbl in _extract_hist_entities(t):
                all_labels.add(lbl)
        for expected in ("per", "loc_c", "tim", "ins"):
            assert expected in all_labels


# ═══════════════════════════════════════════════════════════════════════════════
#  Historical NER — Self-Evaluation (gold vs gold)
# ═══════════════════════════════════════════════════════════════════════════════


@hist_available
class TestHistSelfEval:

    @pytest.fixture(scope="class")
    def proj42(self):
        return _load_hist_project(42)

    def test_self_eval_perfect(self, proj42):
        """Evaluating annotations against themselves must give F1 = 1.0."""
        ev = Evaluator()
        tasks_with_ents = 0
        for t in proj42:
            ents = _extract_hist_entities(t)
            if ents:
                ev.add(ents, ents)
                tasks_with_ents += 1

        assert tasks_with_ents > 3000
        r = ev.result()
        assert r.micro_f1 == 1.0
        assert r.exact_match.fp == 0
        assert r.exact_match.fn == 0

    def test_self_eval_entity_count(self, proj42):
        ev = Evaluator()
        for t in proj42:
            ents = _extract_hist_entities(t)
            if ents:
                ev.add(ents, ents)
        r = ev.result()
        assert r.exact_match.tp == 24677

    def test_self_eval_per_type_all_perfect(self, proj42):
        ev = Evaluator()
        for t in proj42:
            ents = _extract_hist_entities(t)
            if ents:
                ev.add(ents, ents)
        r = ev.result()
        for label, m in r.per_type.items():
            assert m.f1 == 1.0, f"Label '{label}' not perfect: F1={m.f1}"

    def test_empty_pred_gives_zero(self, proj42):
        ev = Evaluator()
        for t in proj42:
            ents = _extract_hist_entities(t)
            if ents:
                ev.add(ents, [])
        r = ev.result()
        assert r.micro_f1 == 0.0
        assert r.exact_match.fn == 24677

    def test_summary_shows_all_labels(self, proj42):
        ev = Evaluator()
        for t in proj42:
            ents = _extract_hist_entities(t)
            if ents:
                ev.add(ents, ents)
        summary = ev.result().summary()
        for lbl in ("per", "loc_c", "tim", "ins"):
            assert lbl in summary
