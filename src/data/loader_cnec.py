import re
from bs4 import BeautifulSoup

class LoaderCnec:
    """
    Loader for CNEC dataset using span-based BIOES tagging.
    """

    def __init__(self, path: str):
        self.path = path

    def load(self):
        sentences = []

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("<doc") or line.startswith("</doc"):
                    continue

                tokens, labels = self._parse_line(line)
                if tokens:
                    sentences.append((tokens, labels))

        return sentences

    # ---------------------------
    # MAIN PARSER (SPAN-BASED)
    # ---------------------------
    def _parse_line(self, line: str):
        line = f"<root>{line}</root>"
        soup = BeautifulSoup(line, "html.parser")

        # ---- step 1: rebuild clean text + collect entity spans
        clean_text = ""
        spans = []  # (start_char, end_char, type)

        def extract(node):
            nonlocal clean_text

            if isinstance(node, str):
                clean_text += node
                return

            if node.name == "ne":
                ent_type = node.get("type")
                start = len(clean_text)

                for child in node.children:
                    extract(child)

                end = len(clean_text)
                spans.append((start, end, ent_type))
                return

            for child in node.children:
                extract(child)

        extract(soup)

        # ---- step 2: tokenize full sentence with offsets
        tokens = []
        token_spans = []

        for m in re.finditer(r"\w+|[^\w\s]", clean_text, re.UNICODE):
            tokens.append(m.group())
            token_spans.append((m.start(), m.end()))

        # ---- step 3: assign BIOES labels
        labels = ["O"] * len(tokens)

        def get_token_indices(start, end):
            return [
                i for i, (ts, te) in enumerate(token_spans)
                if not (te <= start or ts >= end)
            ]

        for start, end, ent_type in spans:
            idxs = get_token_indices(start, end)

            if not idxs:
                continue

            if len(idxs) == 1:
                labels[idxs[0]] = f"S-{ent_type}"
            else:
                labels[idxs[0]] = f"B-{ent_type}"
                for i in idxs[1:-1]:
                    labels[i] = f"I-{ent_type}"
                labels[idxs[-1]] = f"E-{ent_type}"

        # safety check
        assert len(tokens) == len(labels), "Misalignment detected!"

        return tokens, labels