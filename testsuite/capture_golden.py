"""(Re)generate testsuite/golden/*.json from the CURRENT code - the
reference numbers test_examples.py checks future runs against.

Run this deliberately, right after you've verified (by other means: reading
the diff, checking a plot, comparing to the closed-form theory already
built into each example) that a change to the physics/numerics is correct.
Do NOT run it just to make a failing test pass - that defeats the point of
having a regression suite. Re-run individual examples with mos_main.py/
main.py/mos_poly_sweep.py and eyeball their printed output and plots first.

Usage: python3 testsuite/capture_golden.py [example_name ...]
       (with no arguments, regenerates all four)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXAMPLES

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")


def main():
    names = sys.argv[1:] or list(EXAMPLES.keys())
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for name in names:
        if name not in EXAMPLES:
            raise SystemExit(f"Unknown example {name!r} - choices: {list(EXAMPLES.keys())}")
        print(f"Running {name}...")
        metrics = EXAMPLES[name]()
        path = os.path.join(GOLDEN_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
