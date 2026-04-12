import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import classification_report, confusion_matrix
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1
from src.data.mapper import Mapper


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
# Batched prediction
# -----------------------------
def predict_batch(batch_tokens, tokenizer, model, id2label, device):
    enc = tokenizer(
        batch_tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    batch_word_ids = [enc.word_ids(i) for i in range(len(batch_tokens))]
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)

    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1).cpu().tolist()

    batch_results = []

    for sent_preds, word_ids in zip(preds, batch_word_ids):
        sent_labels = []
        last_word = -1

        for pred_id, word_id in zip(sent_preds, word_ids):
            if word_id is None:
                continue
            if word_id != last_word:
                sent_labels.append(id2label[pred_id])
                last_word = word_id

        batch_results.append(sent_labels)

    return batch_results


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="stulcrad/CNEC_2_0_robeczech-base")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    ).to(device)

    id2label = model.config.id2label
    data = load_dataset(args.data)

    total = 0
    correct = 0

    all_gold = []
    all_pred = []

    gold_bio_all = []
    pred_bio_all = []

    print("\n=== TOKEN‑LEVEL COMPARISON ===\n")

    # -----------------------------
    # Batching loop
    # -----------------------------
    for i in range(0, len(data), args.batch_size):
        batch = data[i : i + args.batch_size]
        batch_tokens = [item["tokens"] for item in batch]

        batch_preds = predict_batch(batch_tokens, tokenizer, model, id2label, device)

        for item, pred in zip(batch, batch_preds):
            tokens = item["tokens"]
            gold = item["ner_tags"]

            print(f"Sentence ID {item['id']}")

            gold_norm_seq = []
            pred_norm_seq = []

            for t, g, p in zip(tokens, gold, pred):
                g_norm = normalize_gold(g)
                p_norm = normalize_pred(p)

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

    labels = sorted(set(all_gold) | set(all_pred))
    cm = confusion_matrix(all_gold, all_pred, labels=labels)

    print("Labels:", labels)
    print("Confusion matrix:")
    for row_label, row in zip(labels, cm):
        print(f"{row_label:10} {row}")

    # -----------------------------
    # Span-level metrics
    # -----------------------------
    print("\n=== SPAN‑LEVEL METRICS (seqeval) ===\n")
    print(seqeval_report(gold_bio_all, pred_bio_all, digits=4))
    print("Span‑level F1:", seqeval_f1(gold_bio_all, pred_bio_all))


if __name__ == "__main__":
    main()
