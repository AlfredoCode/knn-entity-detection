import json
import re
import os
import urllib.parse
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict, Any, Union


class LoaderHistorical:
    """
    @brief Loader for Label Studio historical NER annotations.

    Accepts a single path or a list of paths to JSONs.
    Loads all projects, merges them, and groups them by the source text filename.
    Populates:
    - self.single_annotator_tasks
    - self.multi_annotator_tasks
    - self.unannotated_tasks
    """

    _URL_PATH_RE = re.compile(r'\?d=historical_ner/(.+)')
    _TOKEN_RE = re.compile(r'\w+|[^\w\s]', re.UNICODE)

    def __init__(self, paths: Union[str, List[str]]):
        if isinstance(paths, str):
            self.paths = [paths]
        else:
            self.paths = paths
            
        self.single_annotator_tasks: List[Dict[str, Any]] = []
        self.multi_annotator_tasks: List[Dict[str, Any]] = []
        self.unannotated_tasks: List[Dict[str, Any]] = []
        
        self._has_categorized = False

    def _get_clean_filename(self, raw_url: str) -> str:
        """
        Extracts the text filename from the Label Studio data field 
        using string splitting.
        """
        if not raw_url:
            return ""
            
        # Remove URL encoding
        decoded_str = urllib.parse.unquote(raw_url)
        
        # Label Studio stores local paths in the '?d=' parameter
        if "?d=" in decoded_str:
            target_path = decoded_str.split("?d=")[-1].split("&")[0]
        else:
            target_path = decoded_str.split("?")[0]
            
        normalized_path = target_path.replace('\\', '/')
        clean_name = os.path.basename(normalized_path)
        
        return clean_name or decoded_str

    def _trim_span(self, text: str, start: int, end: int) -> Tuple[int, int]:
        """
        Recalculates the start and end offsets to remove leading 
        and trailing whitespaces from the annotated span.
        """
        span_text = text[start:end]
        if not span_text:
            return start, end
            
        # Calculate how many spaces are at the beginning
        l_strip_len = len(span_text) - len(span_text.lstrip())
        # Calculate how many spaces are at the end
        r_strip_len = len(span_text) - len(span_text.rstrip())
        
        new_start = start + l_strip_len
        new_end = end - r_strip_len
        
        return new_start, new_end

    def _categorize_tasks(self):
        """Loads all JSON files and groups them by the text filename."""
        
        grouped_items = defaultdict(list)
        
        for path in self.paths:
            base_dir = os.path.dirname(os.path.abspath(path))
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                item['_base_dir'] = base_dir
                item['_source_file'] = os.path.basename(path)

                valid_annotations = [
                    ann for ann in item.get('annotations', [])
                    if not ann.get('was_cancelled', False)
                ]
                item['valid_annotations'] = valid_annotations
                
                if not valid_annotations:
                    self.unannotated_tasks.append(item)
                    continue

                url = item.get('data', {}).get('text', '')
                src_id = self._get_clean_filename(url)
                grouped_items[src_id].append(item)
                
        # Process grouped texts to determine the actual number of unique annotators
        for src_id, items in grouped_items.items():
            all_annotators = set()
            all_valid_anns = []
            
            # Merge annotators and annotations from all tasks referencing this single text
            for item in items:
                all_valid_anns.extend(item['valid_annotations'])
                all_annotators.update(
                    a.get('completed_by') for a in item['valid_annotations'] if a.get('completed_by') is not None
                )
            
            count = len(all_annotators)
            
            # Create a single item for this text
            representative_item = items[0].copy()
            representative_item['valid_annotations'] = all_valid_anns
            
            if count == 1:
                self.single_annotator_tasks.append(representative_item)
            elif count > 1:
                self.multi_annotator_tasks.append(representative_item)
                
        self._has_categorized = True

    def load(self) -> List[Tuple[List[str], List[str]]]:
        """Loads and processes standard single-annotator data."""
        if not self._has_categorized:
            self._categorize_tasks()

        if self.multi_annotator_tasks or self.single_annotator_tasks:
            self._print_overlap_summary()

        sentences = []
        for item in self.single_annotator_tasks:
            annotation = item['valid_annotations'][0]
            result = self._process_task(item, annotation)
            if result is not None:
                sentences.append(result)

        return sentences

    def load_gold_standard(self) -> List[Tuple[List[str], List[str]]]:
        """
        Loads multi-annotator data and resolves conflicts using Majority Vote.
        Returns a single definitive Gold Standard set for evaluation.
        """
        if not self._has_categorized:
            self._categorize_tasks()

        gold_sentences = []
        for item in self.multi_annotator_tasks:
            result = self._process_multi_task_gold(item)
            if result is not None:
                gold_sentences.append(result)

        return gold_sentences

    def _print_overlap_summary(self):
        counts_by_size = Counter()
        counts_by_size[1] = len(self.single_annotator_tasks)
        group_counts = Counter()

        for item in self.multi_annotator_tasks:
            anns = item.get('valid_annotations', [])
            annotators = tuple(sorted({a.get('completed_by') for a in anns if a.get('completed_by') is not None}))
            counts_by_size[len(annotators)] += 1
            group_counts[annotators] += 1

        print(f"Total UNIQUE text files loaded: {sum(counts_by_size.values())}")
        print("\nTexts by Number of Annotators:")
        print("+---------------------+-------------+")
        print("| No. of Annotators   | Total Texts |")
        print("+---------------------+-------------+")
        for k in sorted(counts_by_size.keys()):
            label = f"{k} annotator" if k == 1 else f"{k} annotators"
            print(f"| {label:<19} | {counts_by_size[k]:>11} |")
        print("+---------------------+-------------+")

        if group_counts:
            print("\nSpecific Annotator Groups Overlap (IAA data pool):")
            print("+-----------------------------+-------------+")
            print("| Annotator IDs               | Shared Texts|")
            print("+-----------------------------+-------------+")
            for group, count in group_counts.most_common():
                group_str = ", ".join(map(str, group))
                print(f"| {group_str:<27} | {count:>11} |")
            print("+-----------------------------+-------------+\n")

    def _resolve_path(self, url: str, base_dir: str) -> Optional[str]:
        m = self._URL_PATH_RE.search(url)
        if not m:
            return None
        rel = urllib.parse.unquote(m.group(1))
        return os.path.join(base_dir, rel)

    def _process_task(self, item: dict, annotation: dict) -> Optional[Tuple[List[str], List[str]]]:
        """Standard processing for single annotations."""
        url = item.get('data', {}).get('text', '')
        local_path = self._resolve_path(url, item.get('_base_dir', ''))
        if not local_path or not os.path.exists(local_path):
            return None

        # Load text first so we can trim the spans correctly
        with open(local_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if not text.strip():
            return None

        results = annotation.get('result', [])

        for r in results:
            if r.get('type') == 'choices':
                choices = r.get('value', {}).get('choices', [])
                if 'czech' not in choices:
                    return None

        spans = []
        for r in results:
            if r.get('type') == 'labels':
                v = r['value']
                label_list = v.get('labels', [])
                if label_list:
                    # Trim whitespace and adjust indices
                    start, end = self._trim_span(text, v['start'], v['end'])
                    
                    # Prevent empty spans (e.g. if the user only highlighted spaces)
                    if start < end:
                        spans.append((start, end, label_list[0]))

        return self._tokenize_and_align(text, spans)

    def _process_multi_task_gold(self, item: dict) -> Optional[Tuple[List[str], List[str]]]:
        """
        Processes a multi-annotator task. Uses Majority Voting to decide which
        spans to keep. Prints a detailed log of kept/discarded entities.
        """
        url = item.get('data', {}).get('text', '')
        local_path = self._resolve_path(url, item.get('_base_dir', ''))
        if not local_path or not os.path.exists(local_path):
            return None

        # Load text primarily to correctly trim spans and log actual words
        with open(local_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if not text.strip():
            return None

        # Extract and trim spans for each annotator
        annotator_spans = {}
        for ann in item.get('valid_annotations', []):
            ann_id = ann.get('completed_by', 'unknown')
            results = ann.get('result', [])
            
            # Skip if not Czech
            is_czech = True
            for r in results:
                if r.get('type') == 'choices' and 'czech' not in r.get('value', {}).get('choices', []):
                    is_czech = False
                    break
            if not is_czech:
                continue

            spans = []
            for r in results:
                if r.get('type') == 'labels':
                    v = r['value']
                    labels = v.get('labels', [])
                    if labels:
                        # Trim whitespace and adjust indices
                        start, end = self._trim_span(text, v['start'], v['end'])
                        if start < end:
                            spans.append((start, end, labels[0]))
            
            annotator_spans[ann_id] = spans

        if not annotator_spans:
            return None

        # Majority Vote Logic
        num_annotators = len(annotator_spans)
        # Threshold: More than 50%. (e.g., 2 for 2, 2 for 3, 3 for 4)
        majority_threshold = (num_annotators // 2) + 1 
        
        span_counts = Counter()
        for spans in annotator_spans.values():
            span_counts.update(spans)

        kept_spans = []
        discarded_spans = []
        
        for span, count in span_counts.items():
            if count >= majority_threshold:
                kept_spans.append(span)
            else:
                discarded_spans.append((span, count))

        # Tokenize and apply BIOES using ONLY the majority-voted spans
        return self._tokenize_and_align(text, kept_spans)

    def _tokenize_and_align(self, text: str, spans: List[Tuple[int, int, str]]) -> Tuple[List[str], List[str]]:
        """Helper to convert raw text and char-spans into BIOES format."""
        token_list = []
        token_spans = []
        for m in self._TOKEN_RE.finditer(text):
            token_list.append(m.group())
            token_spans.append((m.start(), m.end()))

        if not token_list:
            return [], []

        labels = ['O'] * len(token_list)
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