"""Detection evaluation command line: no implicit downloads or expensive default action."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--dataset", type=Path,
                     help="VisDrone2019-DET-val folder (images/ and annotations/); it must match "
                          "eval/results/dataset_manifest.json file for file")
    run.add_argument("--output-dir", type=Path,
                     help="publish the results here instead of eval/results; must not contain earlier results")
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--run-dir", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from eval import detection_run
    if args.command == "run":
        detection_run.run(args.dataset, args.output_dir)
    else:
        detection_run.aggregate(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
