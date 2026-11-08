"""Uncertainty and significance helpers (no TensorFlow needed).

CLI examples
  python -m recyclebin.stats ci 317 380
  python -m recyclebin.stats compare 317 380 201 380        # unpaired two-proportion z-test
  python -m recyclebin.stats mcnemar results/baseline/predictions.csv results/mobilenetv2/predictions.csv
"""
import sys
from math import sqrt
import pandas as pd
from scipy.stats import binomtest, norm


def wilson_ci(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def two_proportion_z(k1, n1, k2, n2):
    p1, p2, pp = k1 / n1, k2 / n2, (k1 + k2) / (n1 + n2)
    z = (p1 - p2) / sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    return z, 2 * norm.sf(abs(z))


def mcnemar_exact(pred_a_csv, pred_b_csv):
    """Paired exact McNemar test on per-image predictions (files from evaluate.py)."""
    a = pd.read_csv(pred_a_csv).set_index("path")
    b = pd.read_csv(pred_b_csv).set_index("path").loc[a.index]
    ca, cb = a["true"] == a["pred"], b["true"] == b["pred"]
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = only_a + only_b
    p = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    return {"only_A_correct": only_a, "only_B_correct": only_b, "p_value": p}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "ci":
        k, n = map(int, args); lo, hi = wilson_ci(k, n)
        print(f"{k}/{n} = {k/n:.3f}  Wilson 95% CI [{lo:.3f}, {hi:.3f}]")
    elif cmd == "compare":
        k1, n1, k2, n2 = map(int, args); z, p = two_proportion_z(k1, n1, k2, n2)
        print(f"z = {z:.2f}, p = {p:.2e}")
    elif cmd == "mcnemar":
        print(mcnemar_exact(*args))
    else:
        sys.exit(__doc__)
