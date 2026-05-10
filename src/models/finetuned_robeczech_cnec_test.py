import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from src.evaluation.metrics import Evaluator, parse_bioes_tags


MODEL_PATH = "best_model_robe_base"  # or best checkpoint folder
VALID_PATH = "out_datasets/cnec_test.jsonl"


# ----------------------------
# Load dataset
# ----------------------------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


valid_data = load_jsonl(VALID_PATH)


# ----------------------------
# Load model
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


id2label = model.config.id2label


# ----------------------------
# Predict function
# ----------------------------
def predict(tokens):
    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    preds = torch.argmax(logits, dim=2)[0].cpu().numpy()

    word_ids = inputs.word_ids(batch_index=0)

    prev_word = None
    labels = []

    for pred, word_id in zip(preds, word_ids):
        if word_id is None or word_id == prev_word:
            continue
        labels.append(id2label[pred])
        prev_word = word_id

    return labels


# ----------------------------
# Evaluation
# ----------------------------
evaluator = Evaluator()

for ex in valid_data:
    tokens = ex["tokens"]
    gold = ex["ner_tags"]

    pred = predict(tokens)

    # IMPORTANT: align lengths (safety)
    if len(pred) != len(gold):
        pred = pred[:len(gold)]

    gold_ents = parse_bioes_tags(gold)
    pred_ents = parse_bioes_tags(pred)

    evaluator.add(gold_ents, pred_ents)


result = evaluator.result()

print(result.summary())
print("\nMicro F1:", result.micro_f1)