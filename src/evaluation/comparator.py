from __future__ import annotations
import json
from typing import List, Dict, Optional
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

@dataclass
class PairwiseAgreement:
    annotator_a: int
    annotator_b: int
    entity_f1: float
    entity_precision: float
    entity_recall: float
    cohens_kappa: float
    num_tasks: int

def compute_pairwise_agreement(records_a: List[AnnotationRecord], records_b: List[AnnotationRecord], text_lengths: Optional[Dict[int, int]] = None) -> float:
    #TODO
    pass

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

    def compute_iaa(self) -> IAAResult:
        return self._compute_iaa_for_records(self._records)