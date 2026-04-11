from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from collections import defaultdict


@dataclass
class Token:
    """
    @brief Represents a single token in a sentence.

    @param text  The surface form of the token.
    @param idx   The index of the token within the sentence.
    """
    text: str
    idx: int


@dataclass
class Entity:
    """
    @brief Represents a labeled entity span.

    @param start  Start token index of the entity.
    @param end    End token index of the entity (inclusive).
    @param label  Entity label (e.g., PER, LOC, ORG).
    """
    start: int
    end: int
    label: str


@dataclass
class Sentence:
    """
    @brief Represents a sentence consisting of tokens and entity annotations.

    @param tokens    List of Token objects.
    @param entities  List of Entity objects associated with the sentence.
    """
    tokens: List[Token]
    entities: List[Entity] = field(default_factory=list)


class CnecModel:
    """
    @brief Container for CNEC dataset sentences and utilities for exporting
           annotations into various tagging schemes (BIO, BIOES, multilabel, spans).

    Provides:
    - BIO export
    - BIOES export
    - Multi-label export (nested-safe)
    - Span export
    """

    def __init__(self):
        """
        @brief Initializes an empty CNEC model.
        """
        self.sentences: List[Sentence] = []

    def add_sentence(self, tokens: List[str], entities: List[Tuple[int, int, str]]):
        """
        @brief Adds a sentence with tokens and entity spans.

        @param tokens    List of token strings.
        @param entities  List of tuples (start_idx, end_idx, label).
        """
        token_objs = [Token(t, i) for i, t in enumerate(tokens)]
        entity_objs = [Entity(s, e, l) for s, e, l in entities]
        self.sentences.append(Sentence(token_objs, entity_objs))

    def __len__(self):
        """
        @brief Returns number of stored sentences.

        @return Number of sentences.
        """
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Sentence:
        """
        @brief Retrieves a sentence by index.

        @param idx  Sentence index.
        @return Sentence object.
        """
        return self.sentences[idx]

    def to_bio(self, sentence_idx: int) -> List[Tuple[str, str]]:
        """
        @brief Converts a sentence into BIO tagging format.

        @param sentence_idx  Index of the sentence to convert.
        @return List of (token_text, BIO_label) tuples.
        """
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

    def to_bioes(self, sentence_idx: int) -> List[Tuple[str, str]]:
        """
        @brief Converts a sentence into BIOES tagging format.

        @param sentence_idx  Index of the sentence to convert.
        @return List of (token_text, BIOES_label) tuples.
        """
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

    def to_spans(self, sentence_idx: int) -> List[Tuple[int, int, str]]:
        """
        @brief Returns entity spans for a sentence.

        @param sentence_idx  Index of the sentence.
        @return List of (start_idx, end_idx, label) tuples.
        """
        sent = self.sentences[sentence_idx]
        return [(e.start, e.end, e.label) for e in sent.entities]

    def to_sparse_multilabel(self, idx: int):
        """
        @brief Converts a sentence into sparse multi-label format.
               Each token receives a dictionary {label: tag} for all labels.

        @param idx  Sentence index.
        @return List of (token_text, {label: tag}) pairs.
        """
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

    def export_all_bio(self):
        """
        @brief Exports all sentences in BIO format.

        @return List of BIO-tagged sentences.
        """
        return [self.to_bio(i) for i in range(len(self.sentences))]

    def export_all_bioes(self):
        """
        @brief Exports all sentences in BIOES format.

        @return List of BIOES-tagged sentences.
        """
        return [self.to_bioes(i) for i in range(len(self.sentences))]

    def export_all_multilabel(self):
        """
        @brief Exports all sentences in multi-label format.

        @return List of multi-label encoded sentences.
        """
        return [self.to_multilabel(i) for i in range(len(self.sentences))]

    def export_all_spans(self):
        """
        @brief Exports all sentences as span annotations.

        @return List of span lists.
        """
        return [self.to_spans(i) for i in range(len(self.sentences))]

    def print_sentence(self, idx: int):
        """
        @brief Prints a sentence and its entity annotations.

        @param idx  Sentence index.
        """
        sent = self.sentences[idx]
        print(" ".join(t.text for t in sent.tokens))
        for e in sent.entities:
            print(f"  ENTITY: {e.label} [{e.start}, {e.end}]")
