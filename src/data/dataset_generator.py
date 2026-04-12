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

        # Load all HistoricalNER data
        historical_data = []
        for project in self.historical_projects:
            loaded = self._map_historical(LoaderHistorical(project).load(), mapper)
            print(f"Historical {os.path.basename(project)}: {len(loaded):>6} sentences")
            historical_data.extend(loaded)

        print(f"Historical total: {len(historical_data):>6} sentences")

        # Shuffle and split historical data proportionally
        rng = random.Random(seed)
        rng.shuffle(historical_data)

        n = len(historical_data)
        n_train = round(n * train_ratio)
        n_val   = round(n * val_ratio)
        
        hist_train = historical_data[:n_train]
        hist_val   = historical_data[n_train:n_train + n_val]
        hist_test  = historical_data[n_train + n_val:]

        train_data = cnec_train + hist_train
        val_data   = cnec_dtest + hist_val
        test_data  = cnec_etest + hist_test

        print(f"\nFinal split ({train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}):")
        print(f"  train: {len(train_data):>6} sentences  (CNEC {len(cnec_train)} + hist {len(hist_train)})")
        print(f"  val:   {len(val_data):>6} sentences  (CNEC {len(cnec_dtest)} + hist {len(hist_val)})")
        print(f"  test:  {len(test_data):>6} sentences  (CNEC {len(cnec_etest)} + hist {len(hist_test)})")
        print(f"  total: {len(train_data) + len(val_data) + len(test_data):>6} sentences")

        os.makedirs(output_dir, exist_ok=True)
        self._save_jsonl(train_data, f"{output_dir}/train.jsonl")
        self._save_jsonl(val_data,   f"{output_dir}/val.jsonl")
        self._save_jsonl(test_data,  f"{output_dir}/test.jsonl")

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