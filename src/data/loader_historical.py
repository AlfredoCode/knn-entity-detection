import json
import re
import os
import urllib.parse
from typing import List, Tuple, Optional


class LoaderHistorical:
    """
    @brief Loader for Label Studio historical NER annotations.

    Each JSON export from Label Studio contains tasks where:
    - data.text is a URL pointing to a local .txt file with the OCR text
    - annotations[].result contains character-offset entity spans

    The .txt files are resolved relative to the directory containing the JSON
    (i.e. resources/HIstoricalNERData/). German and rejected tasks are skipped.

    For overlapping spans, the longer (outer) span takes priority —
    shorter spans that overlap with already-labelled tokens are skipped.

    Returns list of (tokens, labels) in BIOES format with raw Label Studio tags
    (per, loc_c, loc_n, loc_s, ins, tim, med, obj_a, obj_p, groups, evt, ide, misc, amb).
    Use Mapper.LABEL_STUDIO_TO_INTERNAL to convert to the internal schema.
    """

    _URL_PATH_RE = re.compile(r'\?d=historical_ner/(.+)')
    _TOKEN_RE = re.compile(r'\w+|[^\w\s]', re.UNICODE)

    def __init__(self, path: str):
        self.path = path
        self.base_dir = os.path.dirname(os.path.abspath(path))

    def load(self) -> List[Tuple[List[str], List[str]]]:
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sentences = []
        for item in data:
            result = self._process_task(item)
            if result is not None:
                sentences.append(result)

        return sentences

    def _resolve_path(self, url: str) -> Optional[str]:
        m = self._URL_PATH_RE.search(url)
        if not m:
            return None
        rel = urllib.parse.unquote(m.group(1))
        return os.path.join(self.base_dir, rel)

    def _process_task(self, item: dict) -> Optional[Tuple[List[str], List[str]]]:
        url = item.get('data', {}).get('text', '')
        local_path = self._resolve_path(url)
        if not local_path or not os.path.exists(local_path):
            return None

        # Use the first non-cancelled annotation
        annotation = None
        for ann in item.get('annotations', []):
            if not ann.get('was_cancelled', False):
                annotation = ann
                break

        if annotation is None:
            return None

        results = annotation.get('result', [])

        # Skip German or rejected texts
        for r in results:
            if r.get('type') == 'choices':
                choices = r.get('value', {}).get('choices', [])
                if 'german' in choices or 'reject' in choices:
                    return None

        # Collect entity spans: (char_start, char_end, label)
        spans = []
        for r in results:
            if r.get('type') == 'labels':
                v = r['value']
                label_list = v.get('labels', [])
                if label_list:
                    spans.append((v['start'], v['end'], label_list[0]))

        with open(local_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if not text.strip():
            return None

        # Tokenize and record character positions
        token_list = []
        token_spans = []
        for m in self._TOKEN_RE.finditer(text):
            token_list.append(m.group())
            token_spans.append((m.start(), m.end()))

        if not token_list:
            return None

        labels = ['O'] * len(token_list)

        # Apply longer spans first; skip shorter spans overlapping already-labelled tokens
        sorted_spans = sorted(spans, key=lambda s: s[1] - s[0], reverse=True)

        for char_start, char_end, tag in sorted_spans:
            covered = [
                i for i, (ts, te) in enumerate(token_spans)
                if ts >= char_start and te <= char_end
            ]
            if not covered:
                continue

            if any(labels[i] != 'O' for i in covered):
                continue

            if len(covered) == 1:
                labels[covered[0]] = f'S-{tag}'
            else:
                labels[covered[0]] = f'B-{tag}'
                for idx in covered[1:-1]:
                    labels[idx] = f'I-{tag}'
                labels[covered[-1]] = f'E-{tag}'

        return token_list, labels
