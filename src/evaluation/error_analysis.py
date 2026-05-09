import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.data.mapper import LabelNormalizer
from src.evaluation.metrics import (
    Entity,
    Evaluator,
    TokenLevelEvaluator,
    parse_bioes_tags,
)

# identifies worst sentences
# identifies most common type confusions,
# length-bucket f1 breakdown
# OOV rate vs F1


def _bioes_to_bio(tag: str) -> str:
    if tag == "O" or "-" not in tag:
        return tag
    pref, body = tag.split("-", 1)
    if pref == "S":
        return f"B-{body}"
    if pref == "E":
        return f"I-{body}"
    return tag


def _ents(tags: Sequence[str]) -> List[Entity]:
    return parse_bioes_tags(list(tags)) if tags else []


def _example_f1(gold: Sequence[str], pred: Sequence[str]) -> Optional[float]:
    """Per-example micro F1, or ``None`` for the vacuous empty-vs-empty case"""
    g = _ents(gold)
    p = _ents(pred)
    if not g and not p:
        return None
    ev = Evaluator()
    ev.add(g, p)
    return ev.result().micro_f1


@dataclass
class ExampleError:
    """Worst-case container: one example with mismatched tags"""
    index: int
    f1: float
    tokens: List[str]
    gold: List[str]
    pred: List[str]
    diff_positions: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "f1": round(self.f1, 4),
            "tokens": self.tokens,
            "gold": self.gold,
            "pred": self.pred,
            "diff_positions": self.diff_positions,
        }


@dataclass
class ErrorReport:
    num_examples: int
    overall_span_f1: float
    overall_token_acc: float
    worst_examples: List[ExampleError]
    top_confusions: List[Tuple[str, str, int]]
    length_buckets: Dict[str, Dict[str, float]]
    oov_correlation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "num_examples": self.num_examples,
            "overall_span_f1": round(self.overall_span_f1, 4),
            "overall_token_accuracy": round(self.overall_token_acc, 4),
            "worst_examples": [e.to_dict() for e in self.worst_examples],
            "top_confusions": [
                {"gold": g, "pred": p, "count": c}
                for g, p, c in self.top_confusions
            ],
            "length_buckets": self.length_buckets,
        }
        if self.oov_correlation is not None:
            d["oov_correlation"] = self.oov_correlation
        return d

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def text_summary(self) -> str:
        lines = [
            "Error Analysis",
            "=" * 60,
            f"  examples         : {self.num_examples}",
            f"  span micro F1    : {self.overall_span_f1:.4f}",
            f"  token accuracy   : {self.overall_token_acc:.4f}",
            "",
            "Top confusions (gold → pred):",
        ]
        for g, p, c in self.top_confusions:
            lines.append(f"  {g:>20s} → {p:<20s}  ×{c}")
        lines.append("")
        lines.append("Length-bucket F1:")
        for bucket, info in self.length_buckets.items():
            lines.append(
                f"  {bucket:<20s}  n={int(info['n']):4d}  "
                f"F1={info['f1']:.4f}"
            )
        if self.oov_correlation is not None:
            lines.append("")
            lines.append(
                f"OOV: low-OOV F1={self.oov_correlation['low_oov_f1']:.4f}  "
                f"high-OOV F1={self.oov_correlation['high_oov_f1']:.4f}  "
                f"(threshold OOV-rate={self.oov_correlation['threshold']:.2f})"
            )
        lines.append("")
        lines.append("Worst examples:")
        for e in self.worst_examples:
            lines.append(f"  #{e.index}  F1={e.f1:.4f}  "
                         f"({len(e.diff_positions)} diff tokens)")
        return "\n".join(lines)


def _length_bucket(n: int) -> str:
    if n <= 10:
        return "short (≤10)"
    if n <= 30:
        return "medium (11-30)"
    return "long (>30)"


def analyze(
    examples: Sequence[Dict[str, Any]],
    pred_tags_seq: Sequence[Sequence[str]],
    *,
    gold_label_space: str = "internal",
    top_n_worst: int = 20,
    top_k_confusions: int = 10,
    train_vocab: Optional[Iterable[str]] = None,
) -> ErrorReport:
    """Compare gold vs predictions and return a structured ErrorReport"""
    if len(examples) != len(pred_tags_seq):
        raise ValueError("example/pred length mismatch")

    norm = LabelNormalizer.from_name(gold_label_space)

    overall_span = Evaluator()
    overall_token = TokenLevelEvaluator()

    per_example: List[Tuple[int, float, List[str], List[str], List[str]]] = []
    confusion: Dict[Tuple[str, str], int] = {}
    bucket_stats: Dict[str, Dict[str, float]] = {}

    for i, (ex, pred) in enumerate(zip(examples, pred_tags_seq)):
        gold = [_bioes_to_bio(t) for t in norm.normalize_bio(ex["ner_tags"])]
        pred = list(pred)
        if len(gold) != len(pred):
            raise ValueError(
                f"example {i}: gold/pred length mismatch "
                f"({len(gold)} vs {len(pred)})"
            )
        overall_span.add(_ents(gold), _ents(pred))
        overall_token.add(gold, pred)

        f1 = _example_f1(gold, pred)
        # Token-level confusion (strip BIO prefix)
        for g, p in zip(gold, pred):
            gc = g.split("-", 1)[-1] if "-" in g else g
            pc = p.split("-", 1)[-1] if "-" in p else p
            if gc != pc:
                confusion[(gc, pc)] = confusion.get((gc, pc), 0) + 1

        # Skip no entities at all cases from the worst-list and length-bucket averaging
        if f1 is None:
            bucket = _length_bucket(len(ex["tokens"]))
            bucket_stats.setdefault(bucket, {"n": 0.0, "f1_sum": 0.0, "scored_n": 0.0})
            bucket_stats[bucket]["n"] += 1
        else:
            per_example.append((i, f1, list(ex["tokens"]), gold, pred))
            bucket = _length_bucket(len(ex["tokens"]))
            info = bucket_stats.setdefault(bucket, {"n": 0.0, "f1_sum": 0.0, "scored_n": 0.0})
            info["n"] += 1
            info["f1_sum"] += f1
            info["scored_n"] += 1

    # Worst examples by F1
    per_example.sort(key=lambda r: (r[1], r[0]))
    worst: List[ExampleError] = []
    for idx, f1, toks, g, p in per_example[:top_n_worst]:
        diffs = [k for k, (gg, pp) in enumerate(zip(g, p)) if gg != pp]
        worst.append(ExampleError(idx, f1, toks, g, p, diffs))

    # Top-k confusions
    top_conf = sorted(confusion.items(), key=lambda r: -r[1])[:top_k_confusions]
    top_conf_list = [(g, p, c) for (g, p), c in top_conf]

    # Finalise length-bucket F1 (avg over scored-only examples)
    length_buckets: Dict[str, Dict[str, float]] = {}
    for b, info in bucket_stats.items():
        scored = info.get("scored_n", info["n"])
        f1 = info["f1_sum"] / scored if scored > 0 else 0.0
        length_buckets[b] = {"n": info["n"], "scored_n": scored, "f1": f1}

    # OOV correlation
    oov_corr: Optional[Dict[str, Any]] = None
    if train_vocab is not None:
        vocab = set(train_vocab)
        per_oov: List[Tuple[float, float]] = []
        for (idx, f1, toks, _g, _p) in per_example:
            if not toks:
                continue
            oov_rate = sum(1 for t in toks if t not in vocab) / len(toks)
            per_oov.append((oov_rate, f1))
        if per_oov:
            per_oov.sort()
            mid = len(per_oov) // 2
            low = per_oov[:mid]
            high = per_oov[mid:]
            mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
            oov_corr = {
                "threshold": per_oov[mid][0] if mid < len(per_oov) else 0.0,
                "low_oov_f1": mean([f for _r, f in low]),
                "high_oov_f1": mean([f for _r, f in high]),
                "low_oov_n": len(low),
                "high_oov_n": len(high),
            }

    return ErrorReport(
        num_examples=len(examples),
        overall_span_f1=overall_span.result().micro_f1,
        overall_token_acc=overall_token.result().accuracy,
        worst_examples=worst,
        top_confusions=top_conf_list,
        length_buckets=length_buckets,
        oov_correlation=oov_corr,
    )


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _build_argparser():
    import argparse
    p = argparse.ArgumentParser(
        description="Error analysis on stored predictions.",
    )
    p.add_argument("--gold", required=True,
                   help="JSONL with {tokens, ner_tags}.")
    p.add_argument("--pred", required=True,
                   help="JSONL with {pred_tags} aligned to gold by line order.")
    p.add_argument("--train", default=None,
                   help="Optional training JSONL for vocab-based OOV stats.")
    p.add_argument("--gold-label-space", default="internal",
                   choices=["internal", "cnec", "label_studio"])
    p.add_argument("--top-worst", type=int, default=20)
    p.add_argument("--top-confusions", type=int, default=10)
    p.add_argument("--out", required=True)
    p.add_argument("--text-out", default=None)
    return p


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    gold = _load_jsonl(args.gold)
    preds = [r["pred_tags"] for r in _load_jsonl(args.pred)]
    train_vocab = None
    if args.train:
        train_vocab = {t for ex in _load_jsonl(args.train) for t in ex["tokens"]}
    rep = analyze(
        gold, preds,
        gold_label_space=args.gold_label_space,
        top_n_worst=args.top_worst,
        top_k_confusions=args.top_confusions,
        train_vocab=train_vocab,
    )
    out = rep.write(args.out)
    print(f"wrote {out}")
    summary = rep.text_summary()
    print(summary)
    if args.text_out:
        Path(args.text_out).write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    main()
