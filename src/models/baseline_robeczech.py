import argparse
import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from src.data.loader_cnec import LoaderCnec


# =========================
# CONFIG
# =========================
MODEL = "ufal/robeczech-base"


# =========================
# BIOES -> BIO
# =========================
def bioes_to_bio(label: str):
    if label.startswith("E-") or label.startswith("I-"):
        return "I-" + label.split("-", 1)[1]
    if label.startswith("S-"):
        return "B-" + label.split("-", 1)[1]
    return label


# =========================
# DATASET
# =========================
class CNECDataset(Dataset):
    def __init__(self, data, tokenizer, label2id):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, labels = self.data[idx]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True
        )

        word_ids = encoding.word_ids()

        aligned_labels = []
        prev = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != prev:
                aligned_labels.append(
                    self.label2id[bioes_to_bio(labels[word_id])]
                )
            else:
                aligned_labels.append(-100)

            prev = word_id

        encoding["labels"] = aligned_labels
        return encoding


# =========================
# LABELS
# =========================
def build_labels(data):
    labels = set()

    for _, y in data:
        for l in y:
            labels.add(bioes_to_bio(l))

    labels = sorted(labels)

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    return label2id, id2label


# =========================
# ALIGN PREDICTIONS
# =========================
def align_predictions(tokens, preds, word_ids, id2label):
    labels = ["O"] * len(tokens)

    prev = None

    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid != prev:
            labels[wid] = id2label[preds[i]]
        prev = wid

    return labels


# =========================
# TRAIN
# =========================
def train(path):
    print("Loading dataset...")
    loader = LoaderCnec(path)
    data = loader.load()
    print("Sentences:", len(data))

    print("Building labels...")
    label2id, id2label = build_labels(data)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    dataset = CNECDataset(data, tokenizer, label2id)

    train_size = int(0.9 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    eval_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    print("Loading model...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir="./robeczech-cnec",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    print("Training...")
    trainer.train()

    print("Saving model...")
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
    print("Sentences:", len(sentences))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "./robeczech-cnec" if os.path.exists("./robeczech-cnec") else MODEL

    print("Loading model:", model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    model.eval()

    id2label = model.config.id2label

    out_path = "baseline_robeczech.log"

    with open(out_path, "w", encoding="utf-8") as f:
        for tokens, _ in sentences:

            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True
            )

            word_ids = encoding.word_ids()
            encoding = {k: v.to(device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = model(**encoding)

            preds = outputs.logits.argmax(-1)[0].tolist()

            labels = align_predictions(tokens, preds, word_ids, id2label)

            for t, l in zip(tokens, labels):
                f.write(f"{t} {l}\n")

            f.write("\n")

    print("Saved to", out_path)


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