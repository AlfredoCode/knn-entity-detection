import argparse
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
import evaluate


def read_conll_token_level(path):
    """
    @brief Reads a token-level CoNLL-style file.

    Expected format per line:
        token1 label1 token2 label2 token3 label3 ...

    Blank lines are ignored. Each line becomes one sentence.

    @param path  Path to the CoNLL-style file.
    @return HuggingFace Dataset with fields:
            - "tokens": list[list[str]]
            - "ner_tags": list[list[str]]
    """
    sentences_tokens = []
    sentences_labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            sent_tokens = []
            sent_labels = []

            for i in range(0, len(parts), 2):
                token = parts[i]
                label = parts[i + 1]

                sent_tokens.append(token)
                sent_labels.append(label)

            sentences_tokens.append(sent_tokens)
            sentences_labels.append(sent_labels)

    return Dataset.from_dict({
        "tokens": sentences_tokens,
        "ner_tags": sentences_labels
    })


def tokenize_batch(examples, tokenizer, label2id):
    """
    @brief Tokenizes a batch of sentences and aligns labels to subword tokens.

    Uses HuggingFace's tokenizer with `is_split_into_words=True`.
    Subword tokens receive label -100 except the first subword of each word.

    @param examples   Batch from HF Dataset containing "tokens" and "ner_tags".
    @param tokenizer  Pretrained tokenizer.
    @param label2id   Mapping from label string to integer ID.

    @return Tokenized batch with added "labels" field.
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=False
    )

    labels_batch = []

    for i in range(len(examples["tokens"])):
        word_ids = tokenized.word_ids(batch_index=i)

        prev_word = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)

            elif word_id != prev_word:
                label_ids.append(label2id[examples["ner_tags"][i][word_id]])

            else:
                label_ids.append(-100)

            prev_word = word_id

        labels_batch.append(label_ids)

    tokenized["labels"] = labels_batch
    return tokenized


metric = evaluate.load("seqeval")


def compute_metrics(p, id2label):
    """
    @brief Computes seqeval metrics for token classification.

    Converts model predictions and gold labels into label strings,
    ignoring positions with label -100.

    @param p          Tuple (predictions, labels) from Trainer.
    @param id2label   Mapping from label ID to label string.

    @return Dictionary with precision, recall, f1, accuracy.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_preds = []
    true_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        temp_p = []
        temp_l = []

        for p_i, l_i in zip(pred_seq, label_seq):
            if l_i == -100:
                continue

            temp_p.append(id2label[p_i])
            temp_l.append(id2label[l_i])

        true_preds.append(temp_p)
        true_labels.append(temp_l)

    results = metric.compute(
        predictions=true_preds,
        references=true_labels
    )

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def main():
    """
    @brief Main training pipeline for Czech NER using HuggingFace Transformers.

    Steps:
    1. Parse CLI arguments.
    2. Load CoNLL-style dataset.
    3. Build label mappings.
    4. Tokenize dataset.
    5. Split into train/test.
    6. Load pretrained model (Czert or RobeCzech).
    7. Train using Trainer API.
    8. Save model + tokenizer.

    Command-line arguments:
        --path   Path to training data.
        --model  "czert" or "robeczech".
        --epochs Number of training epochs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--model", type=str, default="czert", choices=["czert", "robeczech"])
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    model_name = (
        "UWB-AIR/Czert-B-base-cased"
        if args.model == "czert"
        else "ufal/robeczech-base"
    )

    dataset = read_conll_token_level(args.path)

    # labels (flat)
    all_labels = sorted(set(label for seq in dataset["ner_tags"] for label in seq))

    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}

    print("Labels:", label2id)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = dataset.map(
        lambda x: tokenize_batch(x, tokenizer, label2id),
        batched=True
    )

    dataset = dataset.train_test_split(test_size=0.1)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir="./ner_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_steps=50,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, id2label)
    )

    trainer.train()

    trainer.save_model("./ner_model")
    tokenizer.save_pretrained("./ner_model")


if __name__ == "__main__":
    main()
