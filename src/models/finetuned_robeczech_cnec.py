import json
import numpy as np
import torch
from collections import Counter

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import f1_score

from src.data.mapper import LabelNormalizer, Mapper


# =========================================================
# CONFIG
# =========================================================
MODEL_NAME = "stulcrad/CNEC_2_0_robeczech-base"
MAX_LENGTH = 512
STRIDE = 128


# =========================================================
# LOAD DATA
# =========================================================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


train_data = load_jsonl("out_datasets/train.jsonl")
valid_data = load_jsonl("out_datasets/val.jsonl")


# =========================================================
# LOAD MODEL LABEL SPACE
# =========================================================
tmp_model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

label2id = tmp_model.config.label2id
id2label = tmp_model.config.id2label


# =========================================================
# BUILD CNEC → INTERNAL MAPPING (FIXED)
# =========================================================
model_labels = list(label2id.keys())

model_cnec_codes = set(
    lbl.split("-", 1)[1]
    for lbl in model_labels
    if lbl != "O"
)

# frequency-based mapping (IMPORTANT FIX)
counter = Counter()

for cnec in model_cnec_codes:
    internal = Mapper.CNEC_TO_INTERNAL.get(cnec)
    if internal:
        counter[internal][cnec] += 1

CANONICAL_CNEC = {
    internal: counter[internal].most_common(1)[0][0]
    for internal in counter
}

print("Canonical mapping:", CANONICAL_CNEC)


# =========================================================
# NORMALIZER
# =========================================================
normalizer = LabelNormalizer.internal()


def normalize_tags(tags):
    return normalizer.normalize_bio(tags)


# =========================================================
# BIOES → BIO FIXED CONVERSION
# =========================================================
def internal_to_model_bio(tags):
    out = []

    for tag in tags:
        if tag == "O":
            out.append("O")
            continue

        prefix, internal = tag.split("-", 1)
        cnec = CANONICAL_CNEC[internal]

        # FIXED BIOES LOGIC
        if prefix in ("B", "S"):
            out.append(f"B-{cnec}")
        elif prefix in ("I", "E"):
            out.append(f"I-{cnec}")

    return out


# =========================================================
# PREPROCESS
# =========================================================
def preprocess(example):
    internal = normalize_tags(example["ner_tags"])
    model_tags = internal_to_model_bio(internal)

    example["ner_tags"] = model_tags
    return example


train_data = [preprocess(x) for x in train_data]
valid_data = [preprocess(x) for x in valid_data]


# =========================================================
# DATASET
# =========================================================
train_ds = Dataset.from_list(train_data)
valid_ds = Dataset.from_list(valid_data)


# =========================================================
# TOKENIZER
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# =========================================================
# TOKENIZATION + ALIGNMENT (IMPROVED)
# =========================================================
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=MAX_LENGTH,
        is_split_into_words=True,
        stride=STRIDE,
        return_overflowing_tokens=True,
    )

    labels = []
    for i, ner_tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)

        prev_word = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word:
                label_ids.append(label2id[ner_tags[word_idx]])
            else:
                label_ids.append(-100)

            prev_word = word_idx

        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


train_ds = train_ds.map(tokenize_and_align_labels, batched=True)
valid_ds = valid_ds.map(tokenize_and_align_labels, batched=True)


# =========================================================
# MODEL
# =========================================================
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    id2label=id2label,
    label2id=label2id,
)


# =========================================================
# OPTIMIZER
# =========================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)


# =========================================================
# METRICS
# =========================================================
def compute_metrics(p):
    logits, labels = p
    preds = np.argmax(logits, axis=2)

    true_preds = []
    true_labels = []

    for pred_seq, label_seq in zip(preds, labels):
        seq_preds = []
        seq_labels = []

        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue

            seq_preds.append(id2label[p])
            seq_labels.append(id2label[l])

        true_preds.append(seq_preds)
        true_labels.append(seq_labels)

    return {"f1": f1_score(true_labels, true_preds)}


# =========================================================
# TRAINING ARGS
# =========================================================
training_args = TrainingArguments(
    output_dir="./ner_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=torch.cuda.is_available(),
)


# =========================================================
# TRAINER
# =========================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None),
)


# =========================================================
# TRAIN
# =========================================================
trainer.train()


# =========================================================
# SAVE
# =========================================================
trainer.save_model("./best_model_robe_base")
tokenizer.save_pretrained("./best_model_robe_base")