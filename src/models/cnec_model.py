from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from collections import defaultdict


# -------------------------
# Data structures
# -------------------------
@dataclass
class Token:
    text: str
    idx: int


@dataclass
class Entity:
    start: int   # token index
    end: int     # token index (inclusive)
    label: str


@dataclass
class Sentence:
    tokens: List[Token]
    entities: List[Entity] = field(default_factory=list)


# -------------------------
# Main model
# -------------------------
class CnecModel:
    """
    Stores CNEC dataset in span format and provides:
    - BIO export
    - BIOES export
    - MULTI-LABEL export (nested-safe)
    """

    def __init__(self):
        self.sentences: List[Sentence] = []

    # -------------------------
    # Data ingestion
    # -------------------------
    def add_sentence(self, tokens: List[str], entities: List[Tuple[int, int, str]]):
        token_objs = [Token(t, i) for i, t in enumerate(tokens)]
        entity_objs = [Entity(s, e, l) for s, e, l in entities]
        self.sentences.append(Sentence(token_objs, entity_objs))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Sentence:
        return self.sentences[idx]

    # =========================================================
    # 1. BIO (lossy if nested entities exist)
    # =========================================================
    def to_bio(self, sentence_idx: int) -> List[Tuple[str, str]]:
        sent = self.sentences[sentence_idx]
        labels = ["O"] * len(sent.tokens)

        for ent in sent.entities:
            if ent.start == ent.end:
                labels[ent.start] = f"S-{ent.label}"
            else:
                labels[ent.start] = f"B-{ent.label}"
                for i in range(ent.start + 1, ent.end + 1):
                    labels[i] = f"I-{ent.label}"

        return [(t.text, l) for t, l in zip(sent.tokens, labels)]

    # =========================================================
    # 2. BIOES (still single-label, still lossy for nesting)
    # =========================================================
    def to_bioes(self, sentence_idx: int) -> List[Tuple[str, str]]:
        bio = [lab for _, lab in self.to_bio(sentence_idx)]
        tokens = self.sentences[sentence_idx].tokens

        bioes = bio[:]

        for i, label in enumerate(bio):
            if label.startswith("B-"):
                tag = label[2:]
                if i + 1 >= len(bio) or not bio[i + 1].startswith("I-"):
                    bioes[i] = f"S-{tag}"

            elif label.startswith("I-"):
                tag = label[2:]
                if i + 1 >= len(bio) or not bio[i + 1].startswith("I-"):
                    bioes[i] = f"E-{tag}"

        return [(t.text, l) for t, l in zip(tokens, bioes)]

    # =========================================================
    # 3. SPAN FORMAT (ground truth, lossless)
    # =========================================================
    def to_spans(self, sentence_idx: int) -> List[Tuple[int, int, str]]:
        sent = self.sentences[sentence_idx]
        return [(e.start, e.end, e.label) for e in sent.entities]

    # =========================================================
    # 4. MULTI-LABEL (IMPORTANT: preserves nesting)
    # =========================================================
    def to_sparse_multilabel(self, idx: int):
        sent = self.sentences[idx]
        tokens = sent.tokens

        # collect all labels in dataset
        all_labels = set(
            ent.label
            for s in self.sentences
            for ent in s.entities
        )

        layers = [
            {lab: "O" for lab in all_labels}
            for _ in tokens
        ]

        for ent in sent.entities:
            s, e, label = ent.start, ent.end, ent.label

            if s == e:
                layers[s][label] = "S"
            else:
                layers[s][label] = "B"
                for i in range(s + 1, e):
                    layers[i][label] = "I"
                layers[e][label] = "E"

        return [(t.text, layer) for t, layer in zip(tokens, layers)]

    # =========================================================
    # Export helpers
    # =========================================================
    def export_all_bio(self):
        return [self.to_bio(i) for i in range(len(self.sentences))]

    def export_all_bioes(self):
        return [self.to_bioes(i) for i in range(len(self.sentences))]

    def export_all_multilabel(self):
        return [self.to_multilabel(i) for i in range(len(self.sentences))]

    def export_all_spans(self):
        return [self.to_spans(i) for i in range(len(self.sentences))]

    # =========================================================
    # Debug
    # =========================================================
    def print_sentence(self, idx: int):
        sent = self.sentences[idx]
        print(" ".join(t.text for t in sent.tokens))
        for e in sent.entities:
            print(f"  ENTITY: {e.label} [{e.start}, {e.end}]")