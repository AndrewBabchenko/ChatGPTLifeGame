#!/usr/bin/env python3
"""
Rank evaluation checkpoints using eval_summary.json metrics.

Usage:
  python scripts/rank_eval_results.py --summary outputs/eval_results/eval_summary.json --top 10
  python scripts/rank_eval_results.py --mode predator
"""
import argparse
import json
import math
import sys
from pathlib import Path


DEFAULT_PRED_WEIGHTS = {
    "capture": 0.5,
    "meals": 0.2,
    "starvation": 0.2,
    "time": 0.1,
}

DEFAULT_PREY_WEIGHTS = {
    "escape": 0.5,
    "dist5": 0.2,
    "deaths": 0.2,
    "final": 0.1,
}

DEFAULT_BALANCED_WEIGHTS = {
    "pred": 0.45,
    "prey": 0.45,
    "balance": 0.10,
}


def _normalize(values):
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-9:
        return [0.5] * len(values)
    return [(v - vmin) / (vmax - vmin) for v in values]


def _normalize_inverse(values):
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-9:
        return [0.5] * len(values)
    return [(vmax - v) / (vmax - vmin) for v in values]


def _safe_get(rows, key):
    return [float(r.get(key, 0.0)) for r in rows]


def _compute_scores(rows):
    capture = _safe_get(rows, "predator_capture_rate_mean")
    meals = _safe_get(rows, "predator_meals_per_alive_mean")
    starv = _safe_get(rows, "predator_starvation_deaths_total")
    tcap = _safe_get(rows, "predator_time_to_capture_median")

    escape = _safe_get(rows, "prey_escape_rate_mean")
    dist5 = _safe_get(rows, "prey_dist_gain_5_mean")
    prey_deaths = _safe_get(rows, "prey_deaths_total")
    final_prey = _safe_get(rows, "final_prey_count_mean")
    final_pred = _safe_get(rows, "final_predator_count_mean")

    cap_n = _normalize(capture)
    meals_n = _normalize(meals)
    starv_n = _normalize_inverse(starv)
    tcap_n = _normalize_inverse(tcap)

    escape_n = _normalize(escape)
    dist5_n = _normalize(dist5)
    deaths_n = _normalize_inverse(prey_deaths)
    final_prey_n = _normalize(final_prey)
    final_pred_n = _normalize(final_pred)

    balance_n = [
        math.sqrt(max(0.0, final_prey_n[i] * final_pred_n[i]))
        for i in range(len(rows))
    ]

    for i, row in enumerate(rows):
        pred_score = (
            DEFAULT_PRED_WEIGHTS["capture"] * cap_n[i]
            + DEFAULT_PRED_WEIGHTS["meals"] * meals_n[i]
            + DEFAULT_PRED_WEIGHTS["starvation"] * starv_n[i]
            + DEFAULT_PRED_WEIGHTS["time"] * tcap_n[i]
        )
        prey_score = (
            DEFAULT_PREY_WEIGHTS["escape"] * escape_n[i]
            + DEFAULT_PREY_WEIGHTS["dist5"] * dist5_n[i]
            + DEFAULT_PREY_WEIGHTS["deaths"] * deaths_n[i]
            + DEFAULT_PREY_WEIGHTS["final"] * final_prey_n[i]
        )
        balanced_score = (
            DEFAULT_BALANCED_WEIGHTS["pred"] * pred_score
            + DEFAULT_BALANCED_WEIGHTS["prey"] * prey_score
            + DEFAULT_BALANCED_WEIGHTS["balance"] * balance_n[i]
        )
        row["_pred_score"] = pred_score
        row["_prey_score"] = prey_score
        row["_balanced_score"] = balanced_score


def _print_predator(rows, top_n):
    print("\nTop predator checkpoints:")
    rows = sorted(rows, key=lambda r: r["_pred_score"], reverse=True)
    for r in rows[:top_n]:
        print(
            f"ep{r['checkpoint_episode']:>4} "
            f"score={r['_pred_score']:.3f} "
            f"cap={r.get('predator_capture_rate_mean', 0.0):.3f} "
            f"meals={r.get('predator_meals_per_alive_mean', 0.0):.2f} "
            f"starv={int(r.get('predator_starvation_deaths_total', 0))} "
            f"tcap={r.get('predator_time_to_capture_median', 0.0):.1f}"
        )


def _print_prey(rows, top_n):
    print("\nTop prey checkpoints:")
    rows = sorted(rows, key=lambda r: r["_prey_score"], reverse=True)
    for r in rows[:top_n]:
        print(
            f"ep{r['checkpoint_episode']:>4} "
            f"score={r['_prey_score']:.3f} "
            f"escape={r.get('prey_escape_rate_mean', 0.0):.3f} "
            f"deaths={int(r.get('prey_deaths_total', 0))} "
            f"dist5={r.get('prey_dist_gain_5_mean', 0.0):.2f} "
            f"final={r.get('final_prey_count_mean', 0.0):.1f}"
        )


def _print_balanced(rows, top_n):
    print("\nTop balanced checkpoints:")
    rows = sorted(rows, key=lambda r: r["_balanced_score"], reverse=True)
    for r in rows[:top_n]:
        print(
            f"ep{r['checkpoint_episode']:>4} "
            f"score={r['_balanced_score']:.3f} "
            f"cap={r.get('predator_capture_rate_mean', 0.0):.3f} "
            f"escape={r.get('prey_escape_rate_mean', 0.0):.3f} "
            f"final_prey={r.get('final_prey_count_mean', 0.0):.1f} "
            f"final_pred={r.get('final_predator_count_mean', 0.0):.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Rank eval checkpoints from eval_summary.json")
    parser.add_argument(
        "--summary",
        default="outputs/eval_results/eval_summary.json",
        help="Path to eval_summary.json",
    )
    parser.add_argument("--top", type=int, default=10, help="Number of rows to show")
    parser.add_argument(
        "--mode",
        choices=["predator", "prey", "preys", "balanced", "all"],
        default="all",
        help="Which ranking to print",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"Summary not found: {summary_path}", file=sys.stderr)
        return 1

    data = json.loads(summary_path.read_text())
    rows = data.get("results", [])
    if not rows:
        print("No results in summary file.", file=sys.stderr)
        return 1

    _compute_scores(rows)

    if args.mode in ("predator", "all"):
        _print_predator(rows, args.top)
    if args.mode in ("prey", "preys", "all"):
        _print_prey(rows, args.top)
    if args.mode in ("balanced", "all"):
        _print_balanced(rows, args.top)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
