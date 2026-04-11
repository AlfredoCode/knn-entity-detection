from __future__ import annotations
from typing import Dict, Set, Tuple, List

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
    result = List[str] = []
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