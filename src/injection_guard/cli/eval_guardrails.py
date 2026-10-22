"""``injection-guard-eval``：离线评测命令行入口（P1）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from ..evaluation import evaluate_with_service, load_cases


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 JSONL 用例评测 InjectionGuard")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("eval/golden_cases.jsonl"),
        help="评测用例 JSONL（默认 eval/golden_cases.jsonl）",
    )
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 = 全部）")
    parser.add_argument("--output", type=Path, default=None, help="把指标 JSON 写入文件")
    parser.add_argument("--min-recall", type=float, default=None, help="召回率低于该值时返回非零码")
    parser.add_argument("--dry-run", action="store_true", help="只解析用例，不加载模型")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    cases = load_cases(args.cases)
    if args.limit > 0:
        cases = cases[: args.limit]
    if args.dry_run:
        print(json.dumps({"cases": len(cases), "dry_run": True}, ensure_ascii=False))
        return

    # 惰性导入：--dry-run / 单测不需要 torch 与模型。
    from ..api.dependencies import DefenseService
    from ..config import get_settings

    service = DefenseService(get_settings())
    metrics = evaluate_with_service(service, cases)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")

    if args.min_recall is not None and metrics["recall"] < args.min_recall:
        print(
            f"recall={metrics['recall']:.4f} 低于阈值 {args.min_recall:.4f}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
