from typing import Iterable, List, Optional, Tuple

Entity = Tuple[int, int, str]


class Mapper:
    """Bidirectional mapper between CNEC tags, Label Studio labels, and
    the internal normalized entity schema
    """

    CNEC_TO_INTERNAL = {
        "p": "PersonalName",
        "P": "PersonalName",
        "ps": "PersonalName",
        "pf": "PersonalName",
        "pm": "PersonalName",
        "pd": "PersonalName",
        "pc": "PersonalName",
        "pp": "PersonalName",
        "pb": "PersonalName",
        "p_": "PersonalName",

        "g": "Location_General",
        "g_": "Location_General",
        "gu": "Location_ManMade",
        "gc": "Location_ManMade",
        "gr": "Location_ManMade",
        "gq": "Location_ManMade",

        "gs": "Location_Structure",
        "gp": "Location_Structure",

        "gh": "Location_Natural",
        "gl": "Location_Natural",
        "gt": "Location_Natural",

        "i": "Institution",
        "i_": "Institution",
        "ic": "Institution",
        "if": "Institution",
        "io": "Institution",
        "ia": "Institution",

        "o": "Object",
        "o_": "Object",
        "oa": "Object",
        "op": "Object",
        "om": "Object",
        "oe": "Object",
        "or": "Object",
        "oc": "Object",

        "t": "Time",
        "th": "Time",
        "ty": "Time",
        "tm": "Time",
        "td": "Time",
        "tf": "Time",

        "m": "Media",
        "me": "Media",
        "mn": "Media",
        "ms": "Media",
        "mi": "Media",

        "a": "Address",
        "ah": "Address",
        "at": "Address",
        "az": "Address",
    }

    LABEL_STUDIO_TO_INTERNAL = {
        "per":    "PersonalName",
        "loc_c":  "Location_ManMade",
        "loc_n":  "Location_Natural",
        "loc_s":  "Location_Structure",
        "ins":    "Institution",
        "tim":    "Time",
        "med":    "Media",
        "obj_a":  "Object",
        "obj_p":  "Object",
        "groups": "O",
        "evt":    "O",
        "ide":    "O",
        "misc":   "O",
        "amb":    "O",
    }

    def cnec_to_bioes(self, tokens, entities):
        labels = ["O"] * len(tokens)
        for start, end, cnec_label in entities:
            internal = self.CNEC_TO_INTERNAL.get(cnec_label, "O")
            if internal == "O":
                continue
            length = end - start + 1
            if length == 1:
                labels[start] = f"S-{internal}"
            else:
                labels[start] = f"B-{internal}"
                for i in range(start + 1, end):
                    labels[i] = f"I-{internal}"
                labels[end] = f"E-{internal}"
        return list(zip(tokens, labels))

    def explain_cnec(self, tag: str):
        return self.CNEC_TO_INTERNAL.get(tag, "UNKNOWN")


INTERNAL_LABELS: List[str] = [
    "PersonalName",
    "Location_General",
    "Location_ManMade",
    "Location_Structure",
    "Location_Natural",
    "Institution",
    "Object",
    "Time",
    "Address",
    "Media",
]

INTERNAL_LABEL_SET = set(INTERNAL_LABELS)


class LabelNormalizer:
    """Normalize entity lists from raw CNEC / Label Studio label spacesinto the canonical internal label space"""

    def __init__(self, table: dict, drop_unknown: bool = True) -> None:
        self._table = table
        self._drop_unknown = drop_unknown

    @classmethod
    def cnec(cls) -> "LabelNormalizer":
        return cls(Mapper.CNEC_TO_INTERNAL)

    @classmethod
    def label_studio(cls) -> "LabelNormalizer":
        return cls(Mapper.LABEL_STUDIO_TO_INTERNAL)

    @classmethod
    def internal(cls) -> "LabelNormalizer":
        """Identity normalizer — passes through any internal label unchanged"""
        return cls({lbl: lbl for lbl in INTERNAL_LABELS})

    @classmethod
    def from_name(cls, name: str) -> "LabelNormalizer":
        name = name.lower()
        if name in {"cnec", "cnec_2_0"}:
            return cls.cnec()
        if name in {"label_studio", "ls", "historical"}:
            return cls.label_studio()
        if name in {"internal", "canonical"}:
            return cls.internal()
        raise ValueError(f"Unknown label space: {name!r}")

    def map_label(self, raw: str) -> Optional[str]:
        """Return the internal label, or None to drop the entity"""
        mapped = self._table.get(raw, "O" if self._drop_unknown else raw)
        if mapped == "O":
            return None
        return mapped

    def normalize(self, entities: Iterable[Entity]) -> List[Entity]:
        out: List[Entity] = []
        for start, end, raw in entities:
            mapped = self.map_label(raw)
            if mapped is not None:
                out.append((start, end, mapped))
        return out

    def normalize_bio(self, tags: Iterable[str]) -> List[str]:
        """Normalize a sequence of BIO/BIOES tags into the internal space

        Tags whose entity class maps to "O" become "O"
        Unknown classes (when drop_unknown=False) keep their raw class name
        """
        out: List[str] = []
        for tag in tags:
            if tag == "O" or tag == "":
                out.append("O")
                continue
            if "-" in tag:
                prefix, core = tag.split("-", 1)
            else:
                prefix, core = "", tag
            mapped = self.map_label(core)
            if mapped is None:
                out.append("O")
            else:
                out.append(f"{prefix}-{mapped}" if prefix else mapped)
        return out
