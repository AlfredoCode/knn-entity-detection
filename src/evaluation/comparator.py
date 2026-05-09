from __future__ import annotations
import json
import re
from typing import List, Dict, Optional, Set, Tuple, Any, Hashable
from urllib.parse import unquote, urlparse, parse_qs
from pathlib import Path, PurePosixPath
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import combinations
from src.evaluation.metrics import Entity, Evaluator

SourceId = Hashable

@dataclass
class AnnotationRecord:
    """Annotation for one task"""
    source_id: SourceId
    annotator_id: int
    entities: List[Entity] # List of (start, end, label)
    task_id: Optional[int] = None

@dataclass
class IAAResult:
    """Aggregated IAA across all pairs of annotators"""
    pairwise: List[PairwiseAgreement] = field(default_factory=list)

    @property
    def mean_f1(self) -> float:
        if not self.pairwise:
            return 0.0
        return sum(x.entity_f1 for x in self.pairwise) / len(self.pairwise)

    @property
    def mean_kappa(self) -> float:
        if not self.pairwise:
            return 0.0
        return sum(x.cohens_kappa for x in self.pairwise) / len(self.pairwise)

    def summary(self) -> str:
        lines = ["IAA Summary:"]
        for pair in self.pairwise:
            lines.append(f"Annotator {pair.annotator_a} vs {pair.annotator_b}: F1={pair.entity_f1:.4f}, Kappa={pair.cohens_kappa:.4f} (tasks={pair.num_tasks})")
        lines.append(f"Mean Entity F1: {self.mean_f1:.4f}")
        lines.append(f"Mean Cohen's Kappa: {self.mean_kappa:.4f}")
        return "\n".join(lines)

def _extract_entities_from_result(result: List[Dict[str, Any]]) -> List[Entity]:
    """extracts NER spans from Label Studio format"""
    entities: List[Entity] = []
    for item in result:
        if item.get("type") != "labels":
            continue
        value = item.get("value", {})
        start = value.get("start")
        end = value.get("end")
        labels = value.get("labels", [])
        if start is not None and end is not None and labels:
            for label in labels:
                entities.append((start, end, label))
    return entities

def source_id_from_url(url: str) -> str:
    """Derive a source_id from Label Studio data URL"""
    if not url:
        return ""
    decoded = unquote(url)
    parsed = urlparse(decoded)
    qs = parse_qs(parsed.query)
    if "d" in qs and qs["d"]:
        path = qs["d"][0]
    else:
        path = parsed.path or decoded
    return PurePosixPath(path).name or decoded

def _spans_by_label(entities: List[Entity]) -> Dict[str, Set[int]]:
    """converts list of entities to dict of label"""
    spans: Dict[str, Set[int]] = defaultdict(set)
    for start, end, label in entities:
        spans[label].update(range(start, end))
    return spans

def _cohens_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    """computes Cohen's Kappa from 2x2 matrix"""
    total = tp + fp + fn + tn
    if total == 0:
        return 0.0
    p0 = (tp + tn) / total
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total)
    if abs(1.0 - pe) < 1e-10:
        return 1.0 if abs(p0 - 1.0) < 1e-10 else 0.0
    return (p0 - pe) / (1.0 - pe)
    

def _kappa_for_task(ents_a: List[Entity], ents_b: List[Entity], total_chars: int) -> Tuple[int, int, int, int]:
    """computes TP, FP, FN, TN for one task"""
    spans_a = _spans_by_label(ents_a)
    spans_b = _spans_by_label(ents_b)
    all_labels = set(spans_a) | set(spans_b)
    if not all_labels:
        return 0, 0, 0, 0

    tp = fp = fn = 0
    for label in all_labels:
        span_a = spans_a.get(label, set())
        span_b = spans_b.get(label, set())
        tp += len(span_a & span_b)
        fp += len(span_a - span_b)
        fn += len(span_b - span_a)
    
    tn = total_chars * len(all_labels) - tp - fp - fn
    if tn < 0:
        tn = 0
    return tp, fp, fn, tn

@dataclass
class PairwiseAgreement:
    """Agreement metrics"""
    annotator_a: int
    annotator_b: int
    entity_f1: float
    entity_precision: float
    entity_recall: float
    cohens_kappa: float
    num_tasks: int

def compute_pairwise_agreement(records_a: List[AnnotationRecord], records_b: List[AnnotationRecord], text_lengths: Optional[Dict[int, int]] = None) -> PairwiseAgreement:
    """computes agreement between two sets of annotation records (different annotators, same task), returns PairwiseAgreement"""
    a_by_src: Dict[SourceId, List[Entity]] = {}
    b_by_src: Dict[SourceId, List[Entity]] = {}
    for rec in records_a:
        a_by_src.setdefault(rec.source_id, []).extend(rec.entities)
    for rec in records_b:
        b_by_src.setdefault(rec.source_id, []).extend(rec.entities)

    common = set(a_by_src) & set(b_by_src)
    annotator_a = records_a[0].annotator_id if records_a else -1
    annotator_b = records_b[0].annotator_id if records_b else -1

    if not common:
        return PairwiseAgreement(
            annotator_a=annotator_a, annotator_b=annotator_b,
            entity_f1=0.0, entity_precision=0.0, entity_recall=0.0,
            cohens_kappa=0.0, num_tasks=0,
        )

    eval_ab = Evaluator()
    eval_ba = Evaluator()
    total_tp = total_fp = total_fn = total_tn = 0

    for sid in common:
        ents_a = a_by_src[sid]
        ents_b = b_by_src[sid]
        eval_ab.add(ents_a, ents_b)
        eval_ba.add(ents_b, ents_a)

        if text_lengths and sid in text_lengths:
            total_chars = text_lengths[sid]
        else:
            total_chars = max((e[1] for e in ents_a + ents_b), default=0)

        tp, fp, fn, tn = _kappa_for_task(ents_a, ents_b, total_chars)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    res_ab = eval_ab.result()
    res_ba = eval_ba.result()

    return PairwiseAgreement(
        annotator_a=annotator_a,
        annotator_b=annotator_b,
        entity_f1=(res_ab.micro_f1 + res_ba.micro_f1) / 2,
        entity_precision=(res_ab.micro_precision + res_ba.micro_precision) / 2,
        entity_recall=(res_ab.micro_recall + res_ba.micro_recall) / 2,
        cohens_kappa=_cohens_kappa(total_tp, total_fp, total_fn, total_tn),
        num_tasks=len(common)
    )

def parse_label_studio_json(path: str | Path) -> List[AnnotationRecord]:
    """parses a Label Studio JSON export and returns a list of AnnotationRecords"""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    records: List[AnnotationRecord] = []
    for task in data:
        task_id = task.get("id")
        url = (task.get("data") or {}).get("text") or ""
        sid = source_id_from_url(url) if url else None
        if not sid:
            sid = f"task:{task_id}"
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled", False):
                continue
            annotator_id = ann.get("completed_by")
            if annotator_id is None:
                continue
            entities = _extract_entities_from_result(ann.get("result", []))
            records.append(AnnotationRecord(
                source_id=sid,
                annotator_id=annotator_id,
                entities=entities,
                task_id=task_id
            ))
    return records

class Comparator:
    """computes IAA from Label Studio exports
    
    Usage:
        c = Comparator()
        c.load_project("...json")
        c.load_project("2...json")
        result = c.compute_iaa()
        print(result.summary())
    """
    
    def __init__(self) -> None:
        self._records: List[AnnotationRecord] = []
        self._text_lengths: Dict[int, int] = {}

    def load_project(self, path: str | Path) -> int:
        """loads annotations, returns the number of loaded records"""
        records = parse_label_studio_json(path)
        self._records.extend(records)
        return len(records)
    
    def load_entities_direct(self, source_id: SourceId, annotator_id: int, entities: List[Entity], task_id: Optional[int] = None) -> None:
        """loads entities directly (testing, or non-Label-Studio data)"""
        self._records.append(AnnotationRecord(source_id=source_id, annotator_id=annotator_id,
            entities=entities,
            task_id=task_id))

    def set_text_length(self, task_id: int, length: int) -> None:
        """improves kappa by setting length for a task"""
        self._text_lengths[task_id] = length

    def _compute_iaa_for_records(self, records: List[AnnotationRecord]) -> IAAResult:
        """compute IAA across all annotators (pairs), returns IAAResult"""
        by_annotator: Dict[int, List[AnnotationRecord]] = defaultdict(list)
        for record in records:
            by_annotator[record.annotator_id].append(record)
        
        result = IAAResult()
        for a_id, b_id in combinations(sorted(by_annotator), 2):
            srcs_a = {x.source_id for x in by_annotator[a_id]}
            srcs_b = {x.source_id for x in by_annotator[b_id]}
            if not srcs_a.intersection(srcs_b):
                continue
            agreement = compute_pairwise_agreement(
                by_annotator[a_id], by_annotator[b_id], self._text_lengths,
            )
            if agreement.num_tasks > 0:
                result.pairwise.append(agreement)
        return result

    def compute_iaa(self) -> IAAResult:
        return self._compute_iaa_for_records(self._records)
    
    def per_label_agreement(self) -> Dict[str, IAAResult]:
        """computes IAA separately for each label, returns dict of label -> IAAResult for that label"""
        all_labels: Set[str] = set()
        for rec in self._records:
            for _, _, label in rec.entities:
                all_labels.add(label)

        results: Dict[str, IAAResult] = {}
        for label in sorted(all_labels):
            filtered = [AnnotationRecord(
                    source_id=rec.source_id,
                    annotator_id=rec.annotator_id,
                    entities=[(s, e, l) for s, e, l in rec.entities if l == label],
                    task_id=rec.task_id)
                for rec in self._records]
            results[label] = self._compute_iaa_for_records(filtered)
        return results