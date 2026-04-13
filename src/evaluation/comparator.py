from __future__ import annotations
from typing import List, Dict
from pathlib import Path
from dataclasses import dataclass
from src.evaluation.metrics import Entity

@dataclass
class AnnotationRecord:
    task_id: int
    annotator_id: int
    entities: List[Entity]

@dataclass
class IAAResult:
    #TODO
    pass

def parse_label_studio_json(path: str | Path) -> List[AnnotationRecord]:
    #TODO
    pass

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

    def compute_iaa(self) -> IAAResult:
        #TODO
        pass