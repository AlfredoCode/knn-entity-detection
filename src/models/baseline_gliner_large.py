import argparse
import json
from gliner import GLiNER
from sklearn.metrics import classification_report, confusion_matrix
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1
from src.data.mapper import Mapper
import logging
logging.getLogger("stanza").setLevel(logging.ERROR)

# -----------------------------
# Label normalization
# -----------------------------
def normalize_gold(tag):
    if tag == "O":
        return "O"
    if "-" in tag:
        _, core = tag.split("-", 1)
    else:
        core = tag
    return core


def normalize_pred(tag):
    if tag == "O":
        return "O"
    if "-" in tag:
        _, core = tag.split("-", 1)
    else:
        core = tag
    core = core.lower()
    return Mapper().CNEC_TO_INTERNAL.get(core, "O")


# -----------------------------
# BIO reconstruction for seqeval
# -----------------------------
def to_bio(labels):
    bio = []
    prev = "O"
    for lab in labels:
        if lab == "O":
            bio.append("O")
        else:
            prefix = "B" if prev != lab else "I"
            bio.append(f"{prefix}-{lab}")
        prev = lab
    return bio


# -----------------------------
# Load dataset
# -----------------------------
def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# -----------------------------
# Convert GLiNER spans → token-level labels
# -----------------------------
def spans_to_token_labels(tokens, spans, text):
    pred_labels = ["O"] * len(tokens)

    # compute token offsets in *this* text
    char_offsets = []
    pos = 0
    for tok in tokens:
        start = text.find(tok, pos)
        if start == -1:
            char_offsets.append((None, None))
            continue
        end = start + len(tok)
        char_offsets.append((start, end))
        pos = end

    for span in spans:
        s = span.get("start")
        e = span.get("end")
        if s is None or e is None:
            continue

        label = span["label"]  # use raw GLiNER label here

        for i, (ts, te) in enumerate(char_offsets):
            if ts is None:
                continue
            if not (e <= ts or te <= s):  # overlap
                pred_labels[i] = label

    return pred_labels


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch_size", type=int, default=1)  # GLiNER is not batched
    args = parser.parse_args()

    print("Loading GLiNER model...")
    model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
    model.to("cuda")
    labels = [
        "PersonalName",
        "Location_General",
        "Location_ManMade",
        "Location_Structure",
        "Location_Natural",
        "Institution",
        "Object",
        "Time",
        "Address",
        "Media"
    ]

    data = load_dataset(args.data)

    total = 0
    correct = 0

    all_gold = []
    all_pred = []

    gold_bio_all = []
    pred_bio_all = []

    print("\n=== TOKEN‑LEVEL COMPARISON (GLiNER) ===\n")

    # -----------------------------
    # Loop (pseudo-batching for consistency)
    # -----------------------------
    for i in range(0, len(data), args.batch_size):
        batch = data[i : i + args.batch_size]

        for item in batch:
            tokens = item["tokens"]
            gold = item["ner_tags"]

            text = " ".join(tokens)
            spans = model.predict_entities(text, labels, threshold=0.5)
            pred = spans_to_token_labels(tokens, spans, text)


            print(f"Sentence ID {item['id']}")

            gold_norm_seq = []
            pred_norm_seq = []

            for t, g, p in zip(tokens, gold, pred):
                g_norm = normalize_gold(g)
                p_norm = normalize_gold(p)

                ok = "✓" if g_norm == p_norm else "✗"
                print(f"{t:20} gold={g:20}({g_norm:20}) pred={p:20}({p_norm:20}) {ok}")

                total += 1
                correct += (g_norm == p_norm)

                all_gold.append(g_norm)
                all_pred.append(p_norm)

                gold_norm_seq.append(g)
                pred_norm_seq.append(p_norm)

            print()

            gold_bio_all.append(gold_norm_seq)
            pred_bio_all.append(to_bio(pred_norm_seq))

    print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")

    # -----------------------------
    # Token-level metrics
    # -----------------------------
    print("\n=== TOKEN‑LEVEL METRICS (sklearn) ===\n")
    print(classification_report(all_gold, all_pred, digits=4))

    labels_set = sorted(set(all_gold) | set(all_pred))
    cm = confusion_matrix(all_gold, all_pred, labels=labels_set)

    print("Labels:", labels_set)
    print("Confusion matrix:")
    for row_label, row in zip(labels_set, cm):
        print(f"{row_label:10} {row}")

    # -----------------------------
    # Span-level metrics
    # -----------------------------
    print("\n=== SPAN‑LEVEL METRICS (seqeval) ===\n")
    print(seqeval_report(gold_bio_all, pred_bio_all, digits=4))
    print("Span‑level F1:", seqeval_f1(gold_bio_all, pred_bio_all))


if __name__ == "__main__":
    main()
