import re


class LoaderCnec:
    """
    @brief Loader for the CNEC XML-like annotated dataset.

    This class reads a CNEC-formatted file where named entities are annotated
    using <ne type="..."> ... </ne> tags. It converts each line into a sequence
    of tokens and BIOES labels.
    """

    def __init__(self, path: str):
        """
        @brief Initializes the loader with a file path.

        @param path  Path to the CNEC dataset file.
        """
        self.path = path

    def load(self):
        """
        @brief Loads and parses the entire dataset file.

        Reads the file line by line, skipping XML document tags, and converts
        each annotated line into (tokens, labels) pairs.

        @return List of tuples (tokens, labels), where:
                - tokens: list of token strings
                - labels: list of BIOES labels aligned with tokens
        """
        sentences = []

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # skip XML header
                if not line or line.startswith("<doc") or line.startswith("</doc"):
                    continue

                tokens, labels = self._parse_line(line)
                if tokens:
                    sentences.append((tokens, labels))

        return sentences

    def _parse_line(self, line: str):
        """
        @brief Parses a single annotated line into tokens and BIOES labels.

        Detects <ne type="X"> ... </ne> spans, tokenizes both entity and
        non-entity text, and assigns BIOES labels accordingly.

        @param line  A raw line from the dataset.
        @return (tokens, labels) where:
                - tokens: list of token strings
                - labels: list of BIOES labels
        """
        tokens = []
        labels = []

        pattern = re.compile(r"<ne type=\"(.*?)\">(.*?)</ne>")
        pos = 0

        for match in pattern.finditer(line):
            start, end = match.span()

            # BEFORE ENTITY
            before = line[pos:start]
            before_tokens = self._tokenize(before)

            tokens.extend(before_tokens)
            labels.extend(["O"] * len(before_tokens))

            # ENTITY
            ent_type = match.group(1)
            ent_text = match.group(2)
            ent_tokens = self._tokenize(ent_text)

            if len(ent_tokens) == 1:
                tokens.append(ent_tokens[0])
                labels.append(f"S-{ent_type}")
            else:
                for i, t in enumerate(ent_tokens):
                    tokens.append(t)

                    if i == 0:
                        labels.append(f"B-{ent_type}")
                    elif i == len(ent_tokens) - 1:
                        labels.append(f"E-{ent_type}")
                    else:
                        labels.append(f"I-{ent_type}")

            pos = end

        # TAIL
        tail = line[pos:]
        tail_tokens = self._tokenize(tail)

        tokens.extend(tail_tokens)
        labels.extend(["O"] * len(tail_tokens))

        # sanity check
        assert len(tokens) == len(labels), f"ALIGN ERROR {len(tokens)} vs {len(labels)}"

        return tokens, labels

    def _tokenize(self, text: str):
        """
        @brief Tokenizes text into words and punctuation.

        Uses a regex that splits into:
        - word characters (\\w+)
        - single punctuation characters ([^\\w\\s])

        @param text  Raw text fragment.
        @return List of token strings.
        """
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
