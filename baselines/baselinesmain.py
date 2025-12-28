import argparse
import json

def _cmd_eval_softmax(args: argparse.Namespace) -> None:
    """Run the softmax-based baseline without ATS calibration."""
    from baselines.softmax_baseline.baseline_softmax import evaluate_softmax_baseline

    metrics, csv_path, plot_path = evaluate_softmax_baseline(
        split=args.split, csv_name=args.csv_name
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
    softmax_parser.set_defaults(func=_cmd_eval_softmax)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()