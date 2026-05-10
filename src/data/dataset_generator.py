import json
import random
import os
from src.data.loader_cnec import LoaderCnec
from src.data.loader_historical import LoaderHistorical
from src.data.mapper import Mapper

class DatasetGenerator:
    def __init__(self):
        self.cnec_files = [
            "resources/cnec2.0/data/xml/named_ent_train.xml",
            "resources/cnec2.0/data/xml/named_ent_dtest.xml",
            "resources/cnec2.0/data/xml/named_ent_etest.xml",
        ]
        self.historical_projects = [
            "resources/HIstoricalNERData/project-42-at-2026-03-23-14-04-74326202.json",
            "resources/HIstoricalNERData/project-60-at-2026-03-23-14-36-635a039b.json",
        ]

    def generate_datasets(self, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1.0"

        # Load CNEC data separately
        mapper = Mapper()
        cnec_train = self._map_cnec(LoaderCnec(self.cnec_files[0]).load(), mapper)
        cnec_dtest = self._map_cnec(LoaderCnec(self.cnec_files[1]).load(), mapper)
        cnec_etest = self._map_cnec(LoaderCnec(self.cnec_files[2]).load(), mapper)

        print(f"CNEC train:  {len(cnec_train):>6} sentences")
        print(f"CNEC dtest:  {len(cnec_dtest):>6} sentences")
        print(f"CNEC etest:  {len(cnec_etest):>6} sentences")
        print(f"CNEC total:  {len(cnec_train) + len(cnec_dtest) + len(cnec_etest):>6} sentences")

        # Load HistoricalNER data
        print("\nLoading Historical data...")
        hist_loader = LoaderHistorical(self.historical_projects)
        
        # Single-annotator data (safe for training)
        raw_single = hist_loader.load()
        hist_single = self._map_historical(raw_single, mapper)
        
        # Multi-annotator data
        raw_multi = hist_loader.load_gold_standard()
        hist_multi = self._map_historical(raw_multi, mapper)

        print(f"Historical (Single annotator): {len(hist_single):>6} sentences")
        print(f"Historical (Multi annotator):  {len(hist_multi):>6} sentences")


        # Shuffle and split historical data
        rng = random.Random(seed)
        rng.shuffle(hist_single)
        rng.shuffle(hist_multi)

        total_hist = len(hist_single) + len(hist_multi)
        n_train_target = round(total_hist * train_ratio)
        n_val_target   = round(total_hist * val_ratio)
        
        # The training set is populated with single-annotator data
        n_train = min(n_train_target, len(hist_single))
        hist_train = hist_single[:n_train]

        # The evaluation pool consists of ALL multi-annotator data plus any leftover single-annotator data
        eval_pool = hist_single[n_train:] + hist_multi
        rng.shuffle(eval_pool) # Shuffle thoroughly so validation and test sets get a mix of both groups

        hist_val  = eval_pool[:n_val_target]
        hist_test = eval_pool[n_val_target:]

        print(f"\nCNEC split:")
        print(f"  train: {len(cnec_train):>6} sentences")
        print(f"  val:   {len(cnec_dtest):>6} sentences")
        print(f"  test:  {len(cnec_etest):>6} sentences")

        print(f"\nHistorical split ({train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}):")
        print(f"  train: {len(hist_train):>6} sentences (100% single-annotated)")
        print(f"  val:   {len(hist_val):>6} sentences")
        print(f"  test:  {len(hist_test):>6} sentences")

        os.makedirs(output_dir, exist_ok=True)
        self._save_jsonl(cnec_train, f"{output_dir}/cnec_train.jsonl")
        self._save_jsonl(cnec_dtest, f"{output_dir}/cnec_val.jsonl")
        self._save_jsonl(cnec_etest, f"{output_dir}/cnec_test.jsonl")
        self._save_jsonl(hist_train, f"{output_dir}/historical_train.jsonl")
        self._save_jsonl(hist_val,   f"{output_dir}/historical_val.jsonl")
        self._save_jsonl(hist_test,  f"{output_dir}/historical_test.jsonl")
        print(f"\nSaved to {output_dir}")

    def _map_cnec(self, sentences, mapper):
        records = []
        for tokens, labels in sentences:
            if not tokens or len(tokens) != len(labels):
                continue
            mapped_labels = []
            for t, l in zip(tokens, labels):
                if l == "O":
                    mapped_labels.append("O")
                else:
                    prefix, base = l.split("-", 1)
                    internal = mapper.CNEC_TO_INTERNAL.get(base)
                    mapped_labels.append(f"{prefix}-{internal}" if internal else "O")
            token_strings = [t.text if hasattr(t, "text") else t for t in tokens]
            records.append({"tokens": token_strings, "ner_tags": mapped_labels})
        return records

    def _map_historical(self, sentences, mapper):
        records = []
        for tokens, labels in sentences:
            if not tokens or len(tokens) != len(labels):
                continue
            mapped_labels = []
            for l in labels:
                if l == "O":
                    mapped_labels.append("O")
                else:
                    prefix, base = l.split("-", 1)
                    internal = mapper.LABEL_STUDIO_TO_INTERNAL.get(base, "O")
                    mapped_labels.append(f"{prefix}-{internal}" if internal != "O" else "O")
            records.append({"tokens": tokens, "ner_tags": mapped_labels})
        return records

    def _save_jsonl(self, data, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            for idx, record in enumerate(data):
                entry = {"id": idx, **record}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    generator = DatasetGenerator()
    generator.generate_datasets(output_dir="out_datasets")