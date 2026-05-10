import json
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from seqeval.metrics import f1_score


# =========================================================
# CONFIG
# =========================================================
MODEL_NAME = "stulcrad/CNEC_2_0_robeczech-base"

TRAIN_PATH = "out_datasets/cnec_train.jsonl"
VALID_PATH = "out_datasets/cnec_val.jsonl"

OUTPUT_DIR = "./best_model_robe_base"

MAX_LENGTH = 512

BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 2


# =========================================================
# LOAD JSONL
# =========================================================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


print("Loading datasets...")

train_data = load_jsonl(TRAIN_PATH)
valid_data = load_jsonl(VALID_PATH)

print("Train samples:", len(train_data))
print("Valid samples:", len(valid_data))


# =========================================================
# BUILD LABEL SET FROM DATASET
# =========================================================
print("Building label vocabulary...")

all_labels = set()

for sample in train_data + valid_data:
    for tag in sample["ner_tags"]:
        all_labels.add(tag)

all_labels = sorted(all_labels)

label2id = {label: i for i, label in enumerate(all_labels)}
id2label = {i: label for label, i in label2id.items()}

print("\nLabels:")
for k, v in label2id.items():
    print(f"{v:2d} -> {k}")

print("\nNumber of labels:", len(label2id))


# =========================================================
# TOKENIZER
# =========================================================
print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# =========================================================
# MODEL
# IMPORTANT:
# new classification head
# =========================================================
print("\nLoading model...")

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,

    num_labels=len(label2id),

    id2label=id2label,
    label2id=label2id,

    ignore_mismatched_sizes=True,
)

print("\nModel loaded.")


# =========================================================
# HF DATASETS
# =========================================================
train_ds = Dataset.from_list(train_data)
valid_ds = Dataset.from_list(valid_data)


# =========================================================
# TOKENIZATION + LABEL ALIGNMENT
# =========================================================
def tokenize_and_align_labels(examples):

    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        is_split_into_words=True,
    )

    aligned_labels = []

    for batch_index, labels in enumerate(examples["ner_tags"]):

        word_ids = tokenized.word_ids(batch_index=batch_index)

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:

            # special tokens
            if word_idx is None:
                label_ids.append(-100)

            # first subword
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[labels[word_idx]])

            # remaining subwords ignored
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels

    return tokenized


print("\nTokenizing train dataset...")

train_ds = train_ds.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=train_ds.column_names,
)

print("\nTokenizing valid dataset...")

valid_ds = valid_ds.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=valid_ds.column_names,
)


# =========================================================
# METRICS
# =========================================================
def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):

        current_preds = []
        current_labels = []

        for pred_id, label_id in zip(prediction, label):

            if label_id == -100:
                continue

            current_preds.append(id2label[int(pred_id)])
            current_labels.append(id2label[int(label_id)])

        true_predictions.append(current_preds)
        true_labels.append(current_labels)

    f1 = f1_score(true_labels, true_predictions)

    return {
        "f1": f1
    }


# =========================================================
# TRAINING ARGS
# =========================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=NUM_EPOCHS,

    weight_decay=WEIGHT_DECAY,

    logging_steps=100,

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    save_total_limit=2,

    fp16=torch.cuda.is_available(),

    report_to="none",
)


# =========================================================
# DATA COLLATOR
# =========================================================
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer
)


# =========================================================
# TRAINER
# =========================================================
trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_ds,
    eval_dataset=valid_ds,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)


# =========================================================
# TRAIN
# =========================================================
print("\nStarting training...\n")

trainer.train()


# =========================================================
# SAVE
# =========================================================
print("\nSaving model...\n")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nDone.")