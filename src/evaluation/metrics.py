from __future__ import annotations
from typing import Dict, Set, Tuple

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