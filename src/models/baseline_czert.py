import argparse
import json
import logging

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.metrics import classification_report, confusion_matrix
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1

from src.data.mapper import Mapper

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
        core = core.lower()
        return Mapper().CNEC_TO_INTERNAL.get(core, "O")
    return "O"


# -----------------------------
# BIO reconstruction (seqeval)
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
# Align HF token predictions to whitespace tokens
# -----------------------------
def align_predictions_to_tokens(tokens, text, ner_results):
    pred_labels = ["O"] * len(tokens)

    # build token char offsets (fragile but consistent with your original approach)
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

    # assign predictions
    for ent in ner_results:
        label = ent["entity"]  # e.g. B-PER, I-LOC, etc.
        s, e = ent["start"], ent["end"]

        for i, (ts, te) in enumerate(char_offsets):
            if ts is None:
                continue
            if not (e <= ts or te <= s):  # overlap
                pred_labels[i] = label

    return pred_labels


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model_name", default="UWB-AIR/Czert-B-base-cased")
    args = parser.parse_args()

    # -----------------------------
    # Load model
    # -----------------------------
    print("Loading Czert / CNEC NER model...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name)

    ner_pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="none"  # we want token-level outputs
    )

    # -----------------------------
    # Load data
    # -----------------------------
    data = load_dataset(args.data)

    total = 0
    correct = 0

    all_gold = []
    all_pred = []

    gold_bio_all = []
    pred_bio_all = []

    print("\n=== TOKEN-LEVEL COMPARISON (CZERT NER) ===\n")

    # -----------------------------
    # Evaluation loop
    # -----------------------------
    for item in data:
        tokens = item["tokens"]
        gold = item["ner_tags"]

        text = " ".join(tokens)

        # model inference
        ner_results = ner_pipe(text)

        pred = align_predictions_to_tokens(tokens, text, ner_results)

        print(f"Sentence ID {item.get('id', 'N/A')}")

        gold_norm_seq = []
        pred_norm_seq = []

        for t, g, p in zip(tokens, gold, pred):
            g_norm = normalize_gold(g)
            p_norm = normalize_pred(p)

            ok = "✓" if g_norm == p_norm else "✗"

            print(
                f"{t:20} gold={g:20}({g_norm:20}) "
                f"pred={p:20}({p_norm:20}) {ok}"
            )

            total += 1
            correct += (g_norm == p_norm)

            all_gold.append(g_norm)
            all_pred.append(p_norm)

            gold_norm_seq.append(g)
            pred_norm_seq.append(p_norm)

        print()

        gold_bio_all.append(gold_norm_seq)
        pred_bio_all.append(to_bio(pred_norm_seq))

    # -----------------------------
    # Accuracy
    # -----------------------------
    print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")

    # -----------------------------
    # Token-level metrics
    # -----------------------------
    print("\n=== TOKEN-LEVEL METRICS (sklearn) ===\n")
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
    print("\n=== SPAN-LEVEL METRICS (seqeval) ===\n")
    print(seqeval_report(gold_bio_all, pred_bio_all, digits=4))
    print("Span-level F1:", seqeval_f1(gold_bio_all, pred_bio_all))


if __name__ == "__main__":
    main()