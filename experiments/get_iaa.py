import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.comparator import Comparator

DATA_DIR = ROOT / "resources" / "HIstoricalNERData"
PROJECTS = [
    DATA_DIR / "project-42-at-2026-03-23-14-04-74326202.json",
    DATA_DIR / "project-60-at-2026-03-23-14-36-635a039b.json",
]


def main() -> int:
    missing = [p for p in PROJECTS if not p.exists()]
    if missing:
        print("Missing data files:")
        for m in missing:
            print(f"  {m}")
        return 1

    comp = Comparator()
    total = 0
    for p in PROJECTS:
        n = comp.load_project(p)
        total += n
        print(f"Loaded {n:>5d} records from {p.name}")
    print(f"Total annotation records: {total}")

    annotators = sorted({r.annotator_id for r in comp._records})
    print(f"Annotators: {annotators}")
    print()

    result = comp.compute_iaa()

    print("=" * 60)
    print("INTER-ANNOTATOR AGREEMENT — Historical NER Data")
    print("=" * 60)
    print(f"{'Pair':<14}{'Tasks':>6}{'F1':>9}{'Prec':>9}{'Rec':>9}{'Kappa':>9}")
    print("-" * 60)
    for pa in result.pairwise:
        pair_label = f"{pa.annotator_a} vs {pa.annotator_b}"
        print(
            f"{pair_label:<14}"
            f"{pa.num_tasks:>6}"
            f"{pa.entity_f1:>9.4f}"
            f"{pa.entity_precision:>9.4f}"
            f"{pa.entity_recall:>9.4f}"
            f"{pa.cohens_kappa:>9.4f}"
        )
    print("-" * 60)
    print(f"{'MEAN':<14}{'':>6}{result.mean_f1:>9.4f}{'':>9}{'':>9}{result.mean_kappa:>9.4f}")

    # Per-label agreement, restricted to pairs that actually share data
    print()
    print("=" * 60)
    print("PER-LABEL AGREEMENT (Entity F1, pairs with >=50 shared sources)")
    print("=" * 60)
    per_label = comp.per_label_agreement()
    eligible_pairs = [
        (pa.annotator_a, pa.annotator_b)
        for pa in result.pairwise
        if pa.num_tasks >= 50
    ]
    header = f"{'Label':<10}" + "".join(
        f"{a}v{b:<6}" for a, b in eligible_pairs
    )
    print(header)
    for label in sorted(per_label):
        row = f"{label:<10}"
        label_iaa = per_label[label]
        f1_by_pair = {(p.annotator_a, p.annotator_b): p.entity_f1 for p in label_iaa.pairwise}
        for pair in eligible_pairs:
            f1 = f1_by_pair.get(pair)
            row += f"{f1:>7.3f} " if f1 is not None else f"{'-':>7} "
        print(row)
    return 0


if __name__ == "__main__":
    main()
