import argparse
import os
import json
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)

from seqeval.metrics import f1_score, classification_report

from src.data.loader_cnec import LoaderCnec

MODEL = "ufal/robeczech-base"
MAX_LEN = 512


# =========================
# LOAD DATA
# =========================
def load_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append((obj["tokens"], obj["ner_tags"]))
    return data


# =========================
# LABELS
# =========================
def build_labels(data):
    label_set = set()
    for _, y in data:
        label_set.update(y)

    labels = sorted(label_set)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


# =========================
# DATASET
# =========================
class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, labels = self.data[idx]

        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LEN
        )

        word_ids = enc.word_ids()

        aligned_labels = []
        prev = None

        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev:
                aligned_labels.append(self.label2id[labels[wid]])
            else:
                aligned_labels.append(-100)
            prev = wid

        enc["labels"] = aligned_labels
        return enc


# =========================
# CLASS WEIGHTS (OPTIONAL)
# =========================
def compute_class_weights(data, label2id):
    counts = Counter()

    for _, labels in data:
        for l in labels:
            counts[l] += 1

    total = sum(counts.values())

    weights = np.zeros(len(label2id))

    for label, idx in label2id.items():
        freq = counts[label] / total
        weights[idx] = 1.0 / (np.log(1.2 + freq) + 1e-8)

    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / weights.mean()

    if "O" in label2id:
        weights[label2id["O"]] *= 0.2

    return weights


# =========================
# TRAINER (SIMPLIFIED + STABLE)
# =========================
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            ignore_index=-100
        )

        loss = loss_fct(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss


# =========================
# METRICS
# =========================
def compute_metrics_builder(id2label):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)

        true_preds = []
        true_labels = []

        for pred_seq, label_seq in zip(preds, labels):
            p = []
            l = []

            for pi, li in zip(pred_seq, label_seq):
                if li == -100:
                    continue
                p.append(id2label[int(pi)])
                l.append(id2label[int(li)])

            true_preds.append(p)
            true_labels.append(l)

        return {
            "f1": f1_score(true_labels, true_preds),
            "report": classification_report(true_labels, true_preds)
        }

    return compute_metrics

def allowed_transitions(prev_tag, curr_tag):
    if prev_tag is None:
        return curr_tag.startswith(("O", "B", "S"))

    if prev_tag == "O":
        return curr_tag.startswith(("O", "B", "S"))

    prev_type = prev_tag.split("-", 1)[-1] if "-" in prev_tag else None
    curr_type = curr_tag.split("-", 1)[-1] if "-" in curr_tag else None

    prev_prefix = prev_tag.split("-", 1)[0] if "-" in prev_tag else prev_tag
    curr_prefix = curr_tag.split("-", 1)[0] if "-" in curr_tag else curr_tag

    # valid BIOES transitions
    if prev_prefix == "B":
        return curr_prefix in ["I", "E"] and curr_type == prev_type or curr_prefix == "O"

    if prev_prefix == "I":
        return curr_prefix in ["I", "E"] and curr_type == prev_type or curr_prefix == "O"

    if prev_prefix == "E":
        return curr_prefix in ["O", "B", "S"]

    if prev_prefix == "S":
        return curr_prefix in ["O", "B", "S"]

    return True
# =========================
# WORD-LEVEL DECODING (FIXED)
# =========================
def decode_word_level(word_ids, preds, id2label):
    word_map = defaultdict(list)

    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        word_map[wid].append(preds[i])

    # collapse tokens
    raw = []
    for wid in sorted(word_map.keys()):
        best = Counter(word_map[wid]).most_common(1)[0][0]
        raw.append(id2label[best])

    # enforce BIOES structure
    fixed = []
    prev = None

    for tag in raw:
        if prev is None:
            if tag.startswith(("B", "S", "O")):
                fixed.append(tag)
            else:
                fixed.append("O")
            prev = fixed[-1]
            continue

        if allowed_transitions(prev, tag):
            fixed.append(tag)
        else:
            # repair
            if tag.startswith(("I", "E")):
                fixed.append("B" + tag[1:])
            else:
                fixed.append("O")

        prev = fixed[-1]

    return fixed


# =========================
# TRAIN
# =========================
def train(path):
    print("Loading dataset...")
    data = load_dataset(path)

    print("Label distribution:")
    print(Counter([l for _, y in data for l in y]).most_common(20))

    label2id, id2label = build_labels(data)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    dataset = NERDataset(data, tokenizer, label2id)

    split = int(0.9 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, range(split))
    eval_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    class_weights = compute_class_weights(data, label2id)

    args = TrainingArguments(
        output_dir="./robeczech-cnec",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        class_weights=class_weights,
        compute_metrics=compute_metrics_builder(id2label)
    )

    print("Training...")
    trainer.train()

    trainer.save_model("./robeczech-cnec")
    tokenizer.save_pretrained("./robeczech-cnec")

    print("DONE")


# =========================
# INFERENCE
# =========================
def run(path):
    print("Loading dataset...")
    loader = LoaderCnec(path)
    sentences = loader.load()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "./robeczech-cnec"
    if not os.path.exists(model_path):
        raise ValueError("Model not found. Train first with --train")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    model.eval()

    id2label = model.config.id2label

    out_path = "baseline_robeczech.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, (tokens, _) in enumerate(sentences):

            enc = tokenizer(
                tokens,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LEN
            )

            word_ids = enc.word_ids()

            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                out = model(**enc)

            preds = out.logits.argmax(-1)[0].tolist()

            labels = decode_word_level(word_ids, preds, id2label)

            f.write(json.dumps({
                "id": idx,
                "tokens": tokens,
                "ner_tags": labels
            }, ensure_ascii=False) + "\n")

    print("Saved to", out_path)

def fix_bioes_sequence(tags):
    fixed = []
    prev_type = None
    prev_tag = None

    for i, tag in enumerate(tags):
        if tag == "O":
            fixed.append(tag)
            prev_type = None
            prev_tag = None
            continue

        if "-" not in tag:
            fixed.append("O")
            continue

        prefix, typ = tag.split("-", 1)

        # Start new entity
        if prefix == "S":
            fixed.append(tag)
            prev_type = None
            prev_tag = None

        elif prefix == "B":
            fixed.append(tag)
            prev_type = typ
            prev_tag = "B"

        elif prefix in ["I", "E"]:
            # invalid continuation → fix to B
            if prev_type != typ or prev_tag is None:
                fixed.append("B-" + typ)
                prev_type = typ
                prev_tag = "B"
            else:
                fixed.append(tag)
                prev_tag = prefix

        else:
            fixed.append("O")
            prev_type = None
            prev_tag = None

    return fixed

# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--path", required=True)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if args.train:
        train(args.path)
    else:
        run(args.path)