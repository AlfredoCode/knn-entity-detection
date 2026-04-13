from __future__ import annotations
from typing import List, Dict
from pathlib import Path
from dataclasses import dataclass
from src.evaluation.metrics import Entity, Evaluator

@dataclass
class AnnotationRecord:
    task_id: int
    annotator_id: int
    entities: List[Entity]

@dataclass
class IAAResult:
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
        #TODO
        pass

    def compute_iaa(self) -> IAAResult:
        return self._compute_iaa_for_records(self._records)