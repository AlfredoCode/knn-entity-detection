import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation.metrics import EvaluationResult, TokenLevelMetrics


@dataclass
class RunConfig:
    """Parameters that identify how a run was produced"""
    model: str
    dataset: str
    split: str
    seed: Optional[int] = None
    threshold: Optional[float] = None
    label_space: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "model": self.model,
            "dataset": self.dataset,
            "split": self.split,
            "seed": self.seed,
            "threshold": self.threshold,
            "label_space": self.label_space
        }
        d.update(self.extra)
        return d


def span_result_to_dict(result: EvaluationResult) -> Dict[str, Any]:
    """Serialise a span-level EvaluationResult into a dict"""
    per_label = {}
    for lbl, m in sorted(result.per_type.items()):
        per_label[lbl] = {
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "tp": m.tp,
            "fp": m.fp,
            "fn": m.fn,
            "support": m.support
        }
    return {
        "exact": {
            "precision": result.exact_match.precision,
            "recall": result.exact_match.recall,
            "f1": result.exact_match.f1,
            "tp": result.exact_match.tp,
            "fp": result.exact_match.fp,
            "fn": result.exact_match.fn,
            "support": result.exact_match.support
        },
        "partial": {
            "precision": result.partial_match.precision,
            "recall": result.partial_match.recall,
            "f1": result.partial_match.f1,
            "tp": result.partial_match.tp,
            "fp": result.partial_match.fp,
            "fn": result.partial_match.fn,
            "support": result.partial_match.support
        },
        "micro": {
            "precision": result.micro_precision,
            "recall": result.micro_recall,
            "f1": result.micro_f1
        },
        "macro": {
            "precision": result.macro_precision,
            "recall": result.macro_recall,
            "f1": result.macro_f1
        },
        "per_label": per_label
    }


def token_result_to_dict(result: TokenLevelMetrics) -> Dict[str, Any]:
    """Serialise a TokenLevelMetrics into a dict"""
    labels = result.labels
    per_label: Dict[str, Dict[str, float]] = {}
    for lbl in labels:
        if lbl == "O":
            continue
        m = result.per_label(lbl)
        per_label[lbl] = {
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "tp": m.tp,
            "fp": m.fp,
            "fn": m.fn,
            "support": m.support
        }
    confusion: Dict[str, Dict[str, int]] = {}
    for true_lbl in labels:
        confusion[true_lbl] = {
            pred_lbl: result.confusion.get((true_lbl, pred_lbl), 0)
            for pred_lbl in labels
        }
    return {
        "accuracy": result.accuracy,
        "micro_f1": result.micro_f1,
        "macro_f1": result.macro_f1,
        "labels": labels,
        "per_label": per_label,
        "confusion": confusion
    }


@dataclass
class EvaluationReport:
    """An evaluation report for one run"""
    config: RunConfig
    span: Optional[EvaluationResult] = None
    token: Optional[TokenLevelMetrics] = None
    notes: List[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "config": self.config.to_dict(),
            "notes": list(self.notes)
        }
        if self.span is not None:
            out["span_level"] = span_result_to_dict(self.span)
        if self.token is not None:
            out["token_level"] = token_result_to_dict(self.token)
        return out

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    def text_summary(self) -> str:
        lines = []
        cfg = self.config
        lines.append("=" * 64)
        lines.append(f"Run: model={cfg.model} dataset={cfg.dataset} split={cfg.split}")
        if cfg.threshold is not None:
            lines.append(f"     threshold={cfg.threshold} seed={cfg.seed} label_space={cfg.label_space}")
        lines.append(f"     timestamp={self.timestamp}")
        lines.append("=" * 64)
        if self.span is not None:
            lines.append("\n[span-level]")
            lines.append(self.span.summary())
        if self.token is not None:
            lines.append("\n[token-level]")
            lines.append(self.token.summary())
        if self.notes:
            lines.append("\n[notes]")
            for n in self.notes:
                lines.append(f"- {n}")
        return "\n".join(lines)
