"""Unified CLI for narrative, structured, and Excel evaluation suites."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

from eval.run_eval import excel, narrative, structured
from retrieval.query.lanes import ALL_LANES

SUITES = ("narrative", "structured", "excel")
DEFAULT_LANES = ("narrative", "structured", "excel")


def detect_suite(path: Path) -> str:
    """Infer the adapter from the first non-empty JSONL row."""
    try:
        line = next(line for line in path.read_text().splitlines() if line.strip())
        row = json.loads(line)
    except (OSError, StopIteration, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot inspect evaluation input {path}: {exc}") from exc

    if (
        "plan" in row
        or "expected_channel_behavior" in row
        or row.get("source_scope") in {"excel_only", "cross_corpus"}
    ):
        return "excel"
    evidence = row.get("evidence") or []
    content_types = {
        item.get("content_type") for item in evidence if isinstance(item, dict)
    }
    if content_types.intersection({"table", "figure"}):
        return "structured"
    if row.get("question_type") in {"table", "figure"}:
        return "structured"
    return "narrative"


def feature_slug(value: str) -> str:
    """Preserve useful feature names while preventing path traversal."""
    slug = re.sub(r"\s+", "-", value.strip())
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("._-")
    if not slug:
        raise ValueError("--output must contain at least one letter or number")
    return slug


def output_path(suite: str, run_date: str, feature: str) -> Path:
    filename = f"{run_date}-{feature_slug(feature)}.json"
    if suite == "excel":
        return Path("eval/excel/results") / filename
    return Path("eval/pdf/results") / suite / filename


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a narrative, structured, or Excel JSONL evaluation through "
            "the current retrieval/routing path."
        )
    )
    parser.add_argument(
        "--input",
        "--eval",
        dest="input_path",
        type=Path,
        required=True,
        help="Evaluation JSONL file.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--pdf",
        choices=("narrative", "structured"),
        help="Force the PDF narrative or structured adapter.",
    )
    mode.add_argument(
        "--excel",
        action="store_true",
        help="Force the Excel adapter.",
    )
    parser.add_argument(
        "--output",
        "--feature",
        dest="output_feature",
        required=True,
        help=(
            "User-defined feature label. The evaluator writes it as "
            "YYYY-MM-DD-FEATURE.json in the suite's result directory. "
            "--feature is retained as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Output date in YYYY-MM-DD format; defaults to today.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing dated feature result.",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Use only the suite's own lane instead of current real routing.",
    )
    parser.add_argument(
        "--lanes",
        nargs="+",
        choices=list(ALL_LANES),
        default=None,
        help="Explicit lane set; defaults to narrative+structured+excel.",
    )
    parser.add_argument("--metric-k", type=int, default=None)
    parser.add_argument(
        "--rewrite-mode",
        choices=("auto", "off", "always"),
        default="off",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=30)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument(
        "--embedding-mode",
        choices=("raw", "contextual", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--hybrid-pool-mode",
        choices=("rrf", "union"),
        default="union",
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--misses", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-live-channel", action="store_true")
    parser.add_argument("--allow-corpus-drift", action="store_true")
    parser.add_argument(
        "--grouped",
        action="store_true",
        help=(
            "Evaluate the suite's independently ranked evidence group instead "
            "of the current cross-content merged route."
        ),
    )
    return parser


def _resolve_suite(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> str:
    detected = detect_suite(args.input_path)
    requested = "excel" if args.excel else args.pdf
    if requested and requested != detected:
        parser.error(
            f"{args.input_path} looks like a {detected} suite, "
            f"but --{'excel' if args.excel else 'pdf'} selected {requested}"
        )
    return requested or detected


def _common_args(
    args: argparse.Namespace,
    suite: str,
    destination: Path,
) -> list[str]:
    metric_k = args.metric_k or (10 if suite == "narrative" else 5)
    if args.oracle and args.lanes:
        raise ValueError("--oracle and --lanes cannot be used together")
    lanes = (suite,) if args.oracle else tuple(args.lanes or DEFAULT_LANES)
    common = [
        "--eval",
        str(args.input_path),
        "--metric-k",
        str(metric_k),
        "--out",
        str(destination),
    ]
    if not args.grouped:
        common.extend(["--lanes", *lanes])
    return common


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        parser.error("--date must use YYYY-MM-DD")
    if args.oracle and args.lanes:
        parser.error("--oracle and --lanes cannot be used together")

    suite = _resolve_suite(parser, args)
    destination = output_path(suite, args.date, args.output_feature)
    if destination.exists() and not args.overwrite:
        parser.error(
            f"{destination} already exists; choose another --output or use --overwrite"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    lanes = (suite,) if args.oracle else tuple(args.lanes or DEFAULT_LANES)

    route = "grouped" if args.grouped else "+".join(lanes)
    print(
        "UNIFIED EVAL: "
        f"suite={suite}, input={args.input_path}, "
        f"route={route}, output={destination}"
    )
    runner_args = _common_args(args, suite, destination)
    if suite == "narrative":
        runner_args.extend(
            [
                "--rewrite-mode",
                args.rewrite_mode,
                "--retrieval-top-k",
                str(args.retrieval_top_k),
                "--rerank-top-k",
                str(args.rerank_top_k),
                "--embedding-mode",
                args.embedding_mode,
                "--hybrid-pool-mode",
                args.hybrid_pool_mode,
                "--rrf-k",
                str(args.rrf_k),
            ]
        )
        if args.misses:
            runner_args.append("--misses")
        if args.grouped:
            runner_args.extend(["--content-types", "narrative"])
        narrative.main(runner_args)
    elif suite == "structured":
        runner_args.extend(
            [
                "--rewrite-mode",
                args.rewrite_mode,
                "--retrieval-top-k",
                str(args.retrieval_top_k),
                "--rerank-top-k",
                str(args.rerank_top_k),
            ]
        )
        if args.misses:
            runner_args.append("--misses")
        if args.grouped:
            runner_args.append("--isolate-content-type")
        structured.main(runner_args)
    else:
        if args.limit is not None:
            runner_args.extend(["--limit", str(args.limit)])
        if args.skip_retrieval:
            runner_args.append("--skip-retrieval")
        if args.skip_live_channel:
            runner_args.append("--skip-live-channel")
        if args.allow_corpus_drift:
            runner_args.append("--allow-corpus-drift")
        if args.grouped:
            runner_args.append("--isolate-excel")
        excel.main(runner_args)
