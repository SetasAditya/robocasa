"""One entry point for optional LabApt inspection and validation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from . import ROOT


def main():
    parser = argparse.ArgumentParser(prog="python -m cobot_integration.labapt", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Check geometry and passive dynamics")
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--baseline", type=Path, help="Optional prior scene XML for visual comparison")
    for name, help_text in (
        ("render", "Render the installed layout"),
        ("inventory", "List native task dependencies"),
        ("resets", "Check fixture-compatible task resets"),
        ("rollouts", "Run the five experimental physical workflows"),
    ):
        commands.add_parser(name, help=help_text, add_help=False)
    args, remaining = parser.parse_known_args()
    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run(module, options, **kwargs):
        return subprocess.run(
            [sys.executable, "-m", f"cobot_integration.labapt.{module}", *options],
            cwd=ROOT, env=environment, check=True, **kwargs,
        )

    if args.command == "validate":
        if remaining:
            parser.error("unrecognized arguments: " + " ".join(remaining))
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        with (output / "tests.log").open("w") as log:
            run("_tests", [], stdout=log, stderr=subprocess.STDOUT)
        options = ["--output", str(output / "scene")]
        if args.baseline:
            options += ["--baseline", str(args.baseline.resolve())]
        run("_dynamics", options)
    else:
        module = {
            "render": "_scene", "inventory": "_inventory",
            "resets": "_resets", "rollouts": "_rollouts",
        }[args.command]
        # Resolve caller-relative output/report paths before changing child cwd.
        remaining = list(remaining)
        for index, value in enumerate(remaining[:-1]):
            if value == "--output":
                remaining[index + 1] = str(Path(remaining[index + 1]).resolve())
        if args.command == "render":
            remaining.append("--render")
        run(module, remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
