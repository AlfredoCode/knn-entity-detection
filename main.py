import argparse
from src.data.loader_cnec import LoaderCnec
from src.data.mapper import Mapper

def run(dataset_type: str, dpath: str):
    dataset_type = dataset_type.upper()

    match dataset_type:

        case "CNEC":
            print("Loading CNEC datasets...")

            loader = LoaderCnec(dpath)
            sentences = loader.load()

            print(f"Loaded sentences: {len(sentences)}")

            out_path = "out_cnec.log"
            broken = 0

            mapper = Mapper()

            with open(out_path, "w", encoding="utf-8") as f:
                for tokens, labels in sentences:

                    if not tokens:
                        continue

                    if len(tokens) != len(labels):
                        continue

                    flat = []

                    for t, l in zip(tokens, labels):

                        if l == "O":
                            internal = "O"
                        else:
                            base = l.split("-", 1)[1]  # remove BIOES prefix
                            internal = mapper.CNEC_TO_INTERNAL.get(base, "O")

                        flat.append(t)
                        flat.append(internal)

                    f.write(" ".join(flat) + "\n")

            print(f"BROKEN sentences skipped: {broken}")
            print(f"BIOES dataset written to {out_path}")
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