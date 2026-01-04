import argparse
import json

def _cmd_eval_softmax(args: argparse.Namespace) -> None:
    """Run the softmax-based baseline without ATS calibration."""
    from baselines.softmax_baseline.baseline_softmax import evaluate_softmax_baseline

    metrics, csv_path, plot_path = evaluate_softmax_baseline(
        split=args.split, csv_name=args.csv_name, limit=getattr(args, "limit", None)
    )
    print(f"Saved per-sample probabilities to {csv_path}")
    if plot_path is not None:
        print(f"Saved metric trend plot to {plot_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baselines runner (softmax confidence only; no ATS calibration)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    softmax_parser = subparsers.add_parser(
        "evaluate-softmax", help="Run baseline evaluation using model softmax"
    )
    softmax_parser.add_argument(
        "--csv-name",
        default="baseline_softmax.csv",
        help="Filename for saving per-sample outputs",
    )
    softmax_parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate (train/validation/test)",
    )
    softmax_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of evaluated samples (e.g. 50)",
    )
    softmax_parser.set_defaults(func=_cmd_eval_softmax)

    gen_parser = subparsers.add_parser(
        "generate-only", help="Run generation-only (no confidence scoring)"
    )
    gen_parser.add_argument(
        "--csv-name",
        default="gen_only.csv",
        help="Filename for saving generated outputs",
    )
    gen_parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to generate (train/validation/test)",
    )
    gen_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of generated samples (e.g. 50)",
    )
    gen_parser.set_defaults(func=lambda args: _cmd_gen_only(args))

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


def _cmd_gen_only(args: argparse.Namespace) -> None:
    from baselines.softmax_baseline.baseline_softmax import generate_only

    csv_path = generate_only(split=args.split, csv_name=args.csv_name, limit=getattr(args, "limit", None))
    print(f"Saved generation-only CSV to {csv_path}")


if __name__ == "__main__":
    main()