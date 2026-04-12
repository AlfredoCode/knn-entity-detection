import argparse
from src.data.loader_cnec import LoaderCnec
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
                        internal = f"{prefix}-{mapper.CNEC_TO_INTERNAL.get(base, 'O')}"

                    hf_tokens.append(token)
                    hf_labels.append(internal)

                output.append({
                    "id": idx,
                    "tokens": hf_tokens,
                    "ner_tags": hf_labels
                })

            out_path = "out_cnec.json"

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            print(f"Dataset written to {out_path}")

        case "HISTORICAL":
            # TODO
            pass


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