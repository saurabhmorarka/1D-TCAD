"""Regression tests: rerun each of the four example configurations and
compare their summary metrics against testsuite/golden/*.json.

Run with:  python3 testsuite/test_examples.py
       or: python3 -m unittest discover -s testsuite

A failure means either a real regression (go find and fix the bug - this is
exactly what this suite is for) or an intentional change (rerun
capture_golden.py for that example ONLY after independently verifying the
new numbers are correct, then commit the updated golden/*.json alongside
the code change that caused it).
"""
import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXAMPLES

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")

# Relative tolerance for float comparisons: loose enough to absorb harmless
# cross-platform BLAS/LAPACK noise, tight enough to catch the kind of
# order-of-magnitude/sign/wrong-branch bugs this project has actually hit.
RTOL = 1e-3
ATOL = 1e-8


def _flatten(obj, prefix=""):
    """Yield (path, value) for every leaf (int/float) in a nested dict/list,
    e.g. {"a": [1, 2]} -> [("a[0]", 1), ("a[1]", 2)]."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def _assert_metrics_match(test_case, golden, actual, example_name):
    golden_flat = dict(_flatten(golden))
    actual_flat = dict(_flatten(actual))

    missing = set(golden_flat) - set(actual_flat)
    extra = set(actual_flat) - set(golden_flat)
    test_case.assertFalse(
        missing, f"{example_name}: metrics present in golden but missing from this run: {sorted(missing)}")
    test_case.assertFalse(
        extra, f"{example_name}: metrics present in this run but missing from golden "
               f"(update golden if intentional): {sorted(extra)}")

    mismatches = []
    for key, expected in golden_flat.items():
        got = actual_flat[key]
        if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
            if not math.isclose(got, expected, rel_tol=RTOL, abs_tol=ATOL):
                mismatches.append(f"  {key}: expected {expected!r}, got {got!r}")
        elif got != expected:
            mismatches.append(f"  {key}: expected {expected!r}, got {got!r}")
    test_case.assertFalse(
        mismatches,
        f"{example_name}: {len(mismatches)} metric(s) drifted beyond rtol={RTOL}:\n" + "\n".join(mismatches))


class TestExamplesAgainstGolden(unittest.TestCase):
    """One test method per example, generated below so `python3 -m unittest
    -v` lists them individually (test_diode, test_mos_metal, ...)."""


def _make_test(name, run_fn):
    def test(self):
        golden_path = os.path.join(GOLDEN_DIR, f"{name}.json")
        self.assertTrue(
            os.path.exists(golden_path),
            f"No golden file for {name!r} - run: python3 testsuite/capture_golden.py {name}")
        with open(golden_path) as f:
            golden = json.load(f)
        actual = run_fn()
        _assert_metrics_match(self, golden, actual, name)
    return test


for _name, _run_fn in EXAMPLES.items():
    setattr(TestExamplesAgainstGolden, f"test_{_name}", _make_test(_name, _run_fn))


if __name__ == "__main__":
    unittest.main(verbosity=2)
