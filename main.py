import argparse
import json
from src.data.loader_cnec import LoaderCnec
from src.data.loader_historical import LoaderHistorical
from src.data.mapper import Mapper
import json

def run(dataset_type: str, dpath: str):
    dataset_type = dataset_type.upper()

    match dataset_type:

        case "CNEC":
            print("Loading CNEC datasets...")

            loader = LoaderCnec(dpath)
            sentences = loader.load()

            print(f"Loaded sentences: {len(sentences)}")

            mapper = Mapper()

            output = []

            for idx, (tokens, labels) in enumerate(sentences):

                if not tokens:
                    continue

                if len(tokens) != len(labels):
                    continue

                hf_tokens = []
                hf_labels = []

                for t, l in zip(tokens, labels):

                    token = t.text if hasattr(t, "text") else t

                    if l == "O":
                        internal = "O"
                    else:
                        prefix, base = l.split("-", 1)

                        mapped = mapper.CNEC_TO_INTERNAL.get(base)
                        if mapped is None:
                            internal = "O"
                        else:
                            internal = f"{prefix}-{mapped}"

                    hf_tokens.append(token)
                    hf_labels.append(internal)

                output.append({
                    "id": idx,
                    "tokens": hf_tokens,
                    "ner_tags": hf_labels
                })

            out_path = "out_cnec.jsonl"

            with open(out_path, "w", encoding="utf-8") as f:
                for item in output:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"Dataset written to {out_path}")

        case "HISTORICAL":
            print("Loading Historical NER dataset...")

            loader = LoaderHistorical(dpath)
            sentences = loader.load()

            print(f"Loaded sentences: {len(sentences)}")

            out_path = "out_historical.jsonl"

            mapper = Mapper()

            with open(out_path, "w", encoding="utf-8") as f:
                for idx, (tokens, labels) in enumerate(sentences):

                    if not tokens:
                        continue

                    if len(tokens) != len(labels):
                        continue

                    internal_labels = []
                    for l in labels:
                        if l == "O":
                            internal_labels.append("O")
                        else:
                            prefix, base = l.split("-", 1)
                            internal = mapper.LABEL_STUDIO_TO_INTERNAL.get(base, "O")
                            internal_labels.append(f"{prefix}-{internal}" if internal != "O" else "O")

                    record = {"id": idx, "tokens": tokens, "ner_tags": internal_labels}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"JSON Lines dataset written to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-T", "--type",
        required=True,
        choices=["CNEC", "HISTORICAL"],
        help="Dataset type to load"
    )

    parser.add_argument(
        "-P", "--path",
        required=True,
        help="Path to dataset file or folder"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.type, args.path)