import argparse
from src.data.loader_cnec import LoaderCnec


def run(dataset_type: str, dpath: str):
    dataset_type = dataset_type.upper()

    match dataset_type:

        case "CNEC":
            print("Loading CNEC datasets...")

            loader = LoaderCnec(dpath)
            model = loader.load()

            print("Loaded datasets:", len(model))

            out_path = "out_cnec.log"

            with open(out_path, "w", encoding="utf-8") as f:
                for i in range(len(model)):
                    for token, labels in model.to_bioes(i):
                        f.write(f"{token}\t{labels}\n")
                    f.write("\n")

            print(f"BIO dataset written to {out_path}")

        case "HISTORICAL":
            raise NotImplementedError("HISTORICAL loader not implemented yet")

        case _:
            raise ValueError(f"Unknown dataset type: {dataset_type}")


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
        help="Path to dataset"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.type, args.path)