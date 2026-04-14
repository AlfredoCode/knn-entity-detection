from __future__ import annotations
import json
from typing import List, Dict, Optional, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import combinations
from src.evaluation.metrics import Entity, Evaluator

@dataclass
class AnnotationRecord:
    task_id: int
    annotator_id: int
    entities: List[Entity]

@dataclass
class IAAResult:
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
        #TODO
        pass

def _extract_entities_from_result(result: List[Dict[str, Any]]) -> List[Entity]:
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

def _cohens_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    #TODO
    pass

def _kappa_for_task(ents_a: List[Entity], ents_b: List[Entity], total_chars: int) -> Tuple[int, int, int, int]:
    #TODO
    pass

@dataclass
class PairwiseAgreement:
    annotator_a: int
    annotator_b: int
    entity_f1: float
    entity_precision: float
    entity_recall: float
    cohens_kappa: float
    num_tasks: int

def compute_pairwise_agreement(records_a: List[AnnotationRecord], records_b: List[AnnotationRecord], text_lengths: Optional[Dict[int, int]] = None) -> PairwiseAgreement:
    a_by_task: Dict[int, List[Entity]] = {}
    b_by_task: Dict[int, List[Entity]] = {}
    for rec in records_a:
        a_by_task.setdefault(rec.task_id, []).extend(rec.entities)
    for rec in records_b:
        b_by_task.setdefault(rec.task_id, []).extend(rec.entities)

    common_tasks = set(a_by_task) & set(b_by_task)
    annotator_a = records_a[0].annotator_id if records_a else -1
    annotator_b = records_b[0].annotator_id if records_b else -1
    
    if not common_tasks:
        return PairwiseAgreement(annotator_a=annotator_a, annotator_b=annotator_b, entity_f1=0.0, entity_precision=0.0, entity_recall=0.0, cohens_kappa=0.0, num_tasks=0)

    eval_ab = Evaluator()
    eval_ba = Evaluator()
    total_tp = total_fp = total_fn = total_tn = 0

    for task_id in common_tasks:
        ents_a = a_by_task[task_id]
        ents_b = b_by_task[task_id]
        eval_ab.add(ents_a, ents_b)
        eval_ba.add(ents_b, ents_a)
    
        if text_lengths and task_id in text_lengths:
            total_chars = text_lengths[task_id]
        else:
            total_chars = max((e[1] for e in ents_a + ents_b), default=0)

        tp, fp, fn, tn = _kappa_for_task(ents_a, ents_b, total_chars)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        result_ab = eval_ab.result()
        result_ba = eval_ba.result()

        #TODO might need updates in case of micro/macro f1/prec/rec
        return PairwiseAgreement(annotator_a=annotator_a, annotator_b=annotator_b, entity_f1=(result_ab.f1 + result_ba.f1) / 2, entity_precision=(result_ab.precision + result_ba.precision) / 2, entity_recall=(result_ab.recall + result_ba.recall) / 2, cohens_kappa=_cohens_kappa(total_tp, total_fp, total_fn, total_tn), num_tasks=len(common_tasks))

def parse_label_studio_json(path: str | Path) -> List[AnnotationRecord]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    records: List[AnnotationRecord] = []
    for task in data:
        task_id = task["id"]
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled", False):
                continue
            annotator_id = ann.get("completed_by")
            if annotator_id is None:
                continue
            entities = _extract_entities_from_result(ann.get("result", []))
            records.append(AnnotationRecord(task_id=task_id, annotator_id=annotator_id, entities=entities))
    return records

class Comparator:
    
    def __init__(self) -> None:
        self._records: List[AnnotationRecord] = []
        self._text_lengths: Dict[int, int] = {}

    def load_project(self, path: str | Path) -> int:
        records = parse_label_studio_json(path)
        self._records.extend(records)
        return len(records)
    
    def load_entities_direct(self, task_id: int, annotator_id: int, entities: List[Entity]) -> None:
        self._records.append(AnnotationRecord(task_id=task_id, annotator_id=annotator_id, entities=entities))

    def set_text_length(self, task_id: int, length: int) -> None:
        self._text_lengths[task_id] = length

    def _compute_iaa_for_records(self, records: List[AnnotationRecord]) -> IAAResult:
        by_annotator: Dict[int, List[AnnotationRecord]] = defaultdict(list)
        for record in records:
            by_annotator[record.annotator_id].append(record)
        
        result = IAAResult()
        for a_id, b_id in combinations(sorted(by_annotator), 2):
            tasks_a = {x.task_id for x in by_annotator[a_id]}
            tasks_b = {x.task_id for x in by_annotator[b_id]}
            if not tasks_a.intersection(tasks_b):
                continue
            agreement = compute_pairwise_agreement(by_annotator[a_id], by_annotator[b_id], self._text_lengths)
            if agreement.num_tasks > 0:
                result.pairwise.append(agreement)
        return result

    def compute_iaa(self) -> IAAResult:
        return self._compute_iaa_for_records(self._records)
    
    def per_label_agreement(self) -> Dict[str, IAAResult]:
        all_labels: Set[str] = set()
        for rec in self._records:
            for _, _, label in rec.entities:
                all_labels.add(label)

            results: Dict[str, IAAResult] = {}
            for label in sorted(all_labels):
                filtered = [AnnotationRecord(task_id=rec.task_id, annotator_id=rec.annotator_id, entities=[(s, e, l) for s, e, l in rec.entities if l == label]) for rec in self._records]
                results[label] = self._compute_iaa_for_records(filtered)
        return results