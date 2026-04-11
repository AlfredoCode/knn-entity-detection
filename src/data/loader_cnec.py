import os
import re
import xml.etree.ElementTree as ET

from src.models.cnec_model import CnecModel


class LoaderCnec:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> CnecModel:
        model = CnecModel()

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset path not found: {self.path}")

        files = self._collect_files(self.path)

        for file in files:
            sentences = self._parse_file(file)

            for tokens, entities in sentences:
                model.add_sentence(tokens, entities)

        return model

    # -------------------------
    # File collection
    # -------------------------
    def _collect_files(self, path: str):
        if os.path.isfile(path):
            # single file case
            if path.endswith(".xml") or path.endswith(".sgml"):
                return [path]
            else:
                raise ValueError(f"Unsupported file type: {path}")

        if os.path.isdir(path):
            out = []
            for root, _, files in os.walk(path):
                for f in files:
                    if f.endswith(".xml") or f.endswith(".sgml"):
                        out.append(os.path.join(root, f))
            return out

        raise FileNotFoundError(f"Path not found: {path}")

    # -------------------------
    # Parse file
    # -------------------------
    def _parse_file(self, filepath: str):
        tree = ET.parse(filepath)
        root = tree.getroot()

        tokens, entities = self._parse_document(root)

        if not tokens:
            return []

        return [(tokens, entities)]

    # -------------------------
    # Core XML parsing (NESTING SAFE)
    # -------------------------
    def _parse_document(self, root):
        tokens = []
        entities = []

        buffer = []
        entity_stack = []  # (label, start_token_idx)

        def flush_buffer():
            nonlocal buffer, tokens
            if buffer:
                text = "".join(buffer)
                words = self._tokenize(text)
                tokens.extend(words)
                buffer.clear()

        def walk(node):
            nonlocal buffer, entities

            # normal text
            if node.text:
                buffer.append(node.text)

            for child in node:
                if child.tag == "ne":
                    # flush text before entity starts
                    flush_buffer()

                    label = child.attrib.get("type", "UNK")
                    start_idx = len(tokens)

                    entity_stack.append((label, start_idx))

                    # recurse into nested entity
                    walk(child)

                    # flush inside entity
                    flush_buffer()

                    label, start = entity_stack.pop()
                    end = len(tokens) - 1

                    # VALID SPAN
                    if start <= end:
                        entities.append((start, end, label))

                else:
                    walk(child)

                # tail text (important for XML correctness)
                if child.tail:
                    buffer.append(child.tail)

        walk(root)
        flush_buffer()

        return tokens, entities

    # -------------------------
    # Tokenizer
    # -------------------------
    def _tokenize(self, text: str):
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)