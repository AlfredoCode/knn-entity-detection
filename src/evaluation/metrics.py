from __future__ import annotations
from typing import Dict, Set, Tuple, List, Sequence, Optional
from dataclasses import dataclass, field

Entity = Tuple[int, int, str] # (start, end, label)

# CNEC entity classes
CNEC_CLASSES: Set[str] = {
    "A", "C", "P", "T",
    "ah", "at", "az",
    "g_", "gc", "gh", "gl", "gq", "gr", "gs", "gt", "gu",
    "i_", "ia", "ic", "if", "io",
    "me", "mi", "mn", "ms",
    "n_", "na", "nb", "nc", "ni", "no", "ns",
    "o_", "oa", "oe", "om", "op", "or",
    "p_", "pc", "pd", "pf", "pm", "pp", "ps",
    "td", "tf", "th", "tm", "ty",
}

# Label Studio aliases
STUDIO_LABELS: Set[str] = {
    "per", "loc_n", "loc_c", "loc_s", "ins", "med",
    "obj_a", "obj_p", "tim", "evt", "ide", "groups", "misc", "amb",
}

# Supertype mapping for CNEC
CNEC_SUPERTYPE: Dict[str, str] = {}
for _tag in CNEC_CLASSES:
    if len(_tag) == 1:
        CNEC_SUPERTYPE[_tag] = _tag
    else:
        CNEC_SUPERTYPE[_tag] = _tag[0].upper()

def parse_cnec_entity(entity_str: str) -> List[Entity]:
    entities: List[Entity] = []
    stack: List[Tuple[str, int]] = []  # (tag, start_offset)
    plain_chars: List[str] = []
    i = 0
    entity_len = len(entity_str)
    while i < entity_len:
        entity_char = entity_str[i]
        if entity_char == '<':
            j = i + 1
            tag_start = j
            while j < entity_len and entity_str[j] not in (" ", ">", "<"):
                j += 1
            tag = entity_str[tag_start:j]
            if tag in CNEC_CLASSES or tag in ("cap", "lower", "upper", "segm", "s", "f", "?"):
                stack.append((tag, len(plain_chars)))
                i = j
                if i < entity_len and entity_str[i] == " ":
                    i += 1
            else:
                plain_chars.append(entity_char)
                i += 1
        elif entity_char == ">" and stack:
            tag, start = stack.pop()
            if tag in CNEC_CLASSES:
                entities.append((start, len(plain_chars), tag))
            i += 1
        else:
            plain_chars.append(entity_char)
            i += 1

    return entities

def get_plain_text_cnec(entity_str: str) -> str:
    result: List[str] = []
    i, stack_depth = 0, 0
    entity_len = len(entity_str)
    while i < entity_len:
        entity_char = entity_str[i]
        if entity_char == '<':
            j = i + 1
            while j < entity_len and entity_str[j] not in (" ", ">", "<"):
                j += 1
            tag = entity_str[i+1:j]
            if tag in CNEC_CLASSES or tag in ("cap", "lower", "upper", "segm", "s", "f", "?"):
                stack_depth += 1
                i = j
                if i < entity_len and entity_str[i] == " ":
                    i += 1
                    continue
        if entity_char == ">" and stack_depth > 0:
            stack_depth -= 1
            i += 1
            continue
        result.append(entity_char)
        i += 1
    return "".join(result)

def parse_bioes_tags(tags: Sequence[str]) -> List[Entity]:
    entities: List[Entity] = []
    start: Optional[int] = None
    current_label: Optional[str] = None

    for idx, tag in enumerate(tags):
        if tag == "O" or tag == "o":
            if current_label is not None:
                entities.append((start, idx, current_label))
                start, current_label = None, None
            continue

        parts = tag.split("-", 1)
        if len(parts) != 2:
            if current_label is not None:
                entities.append((start, idx, current_label))
                start, current_label = None, None
            continue

        prefix, label = parts

        if prefix in ("B", "b"):
            if current_label is not None:
                entities.append((start, idx, current_label))
            start, current_label = idx, label

        elif prefix in ("I", "i"):
            if current_label is None or label != current_label:
                if current_label is not None:
                    entities.append((start, idx, current_label))
                start, current_label = idx, label

        elif prefix in ("E", "e"):
            if current_label is not None and label == current_label:
                entities.append((start, idx + 1, current_label))
            else:
                entities.append((idx, idx + 1, label))
            start, current_label = None, None

        elif prefix in ("S", "s"):
            if current_label is not None:
                entities.append((start, idx, current_label))
            entities.append((idx, idx + 1, label))
            start, current_label = None, None

        else:
            if current_label is not None:
                entities.append((start, idx, current_label))
                start, current_label = None, None

    if current_label is not None:
        entities.append((start, len(tags), current_label))

    return entities

def _exact_match(gold: Entity, pred: Entity) -> bool:
    return gold[0] == pred[0] and gold[1] == pred[1] and gold[2] == pred[2]


def _type_match(gold: Entity, pred: Entity) -> bool:
    return gold[2] == pred[2]


def _span_overlap(gold: Entity, pred: Entity) -> bool:
    return gold[0] < pred[1] and pred[0] < gold[1]


def _partial_match(gold: Entity, pred: Entity) -> bool:
    return _span_overlap(gold, pred) and _type_match(gold, pred)

@dataclass
class EntityMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def support(self) -> int:
        return self.tp + self.fn

@dataclass
class EvaluationResult:
    per_type: Dict[str, EntityMetrics] = field(default_factory=dict)
    exact_match: EntityMetrics = field(default_factory=EntityMetrics)
    partial_match: EntityMetrics = field(default_factory=EntityMetrics)

    @property
    def macro_precision(self) -> float:
        types_with_support = [m for m in self.per_type.values() if m.support > 0]
        if not types_with_support:
            return 0.0
        return sum(m.precision for m in types_with_support) / len(types_with_support)

    @property
    def macro_recall(self) -> float:
        types_with_support = [m for m in self.per_type.values() if m.support > 0]
        if not types_with_support:
            return 0.0
        return sum(m.recall for m in types_with_support) / len(types_with_support)

    @property
    def macro_f1(self) -> float:
        types_with_support = [m for m in self.per_type.values() if m.support > 0]
        if not types_with_support:
            return 0.0
        return sum(m.f1 for m in types_with_support) / len(types_with_support)

    @property
    def micro_f1(self) -> float:
        return self.exact_match.f1

    @property
    def micro_precision(self) -> float:
        return self.exact_match.precision

    @property
    def micro_recall(self) -> float:
        return self.exact_match.recall

    def summary(self) -> str:
        lines = []
        lines.append(f"{'Type':<12} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Support':>8}")
        lines.append("-" * 44)

        for label in sorted(self.per_type):
            m = self.per_type[label]
            if m.support > 0 or (m.tp + m.fp) > 0:
                lines.append(
                    f"{label:<12} {m.precision:>7.2%} {m.recall:>7.2%} "
                    f"{m.f1:>7.2%} {m.support:>8d}"
                )

        lines.append("-" * 44)
        lines.append(
            f"{'micro':.<12} {self.micro_precision:>7.2%} {self.micro_recall:>7.2%} "
            f"{self.micro_f1:>7.2%} {self.exact_match.support:>8d}"
        )
        lines.append(
            f"{'macro':.<12} {self.macro_precision:>7.2%} {self.macro_recall:>7.2%} "
            f"{self.macro_f1:>7.2%}"
        )
        lines.append(
            f"{'partial':.<12} {self.partial_match.precision:>7.2%} "
            f"{self.partial_match.recall:>7.2%} {self.partial_match.f1:>7.2%}"
        )
        return "\n".join(lines)