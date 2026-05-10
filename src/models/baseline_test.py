import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)

    id2label = model.config.id2label

    tokens = args.text.split()

    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        out = model(**enc)

    preds = torch.argmax(out.logits, dim=-1)[0]
    word_ids = enc.word_ids()

    seen = set()

    print("\nRESULT:")

    for i, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen:
            continue

        seen.add(word_id)

        label = id2label[int(preds[i])]
        print(f"{tokens[word_id]:15} -> {label}")


if __name__ == "__main__":
    main()