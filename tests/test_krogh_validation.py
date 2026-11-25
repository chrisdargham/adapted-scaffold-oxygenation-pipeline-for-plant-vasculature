"""Check src/solver/krogh_validation.py against the exact zero-order solutions.
Run from the repository root:  python tests/test_krogh_validation.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "solver"))
from krogh_validation import (penetration_depth_from_threshold,  # noqa: E402
                              solve_krogh_mm_radial, solve_steady_slab)

D, V, C0 = 1.5e-9, 1e-4, 0.2
KM_ZERO_ORDER = 1e-9   # Km << C: consumption is effectively zero order

# Slab, oxygen stays positive (L below the critical depth sqrt(2 D C0 / V) = 2.4 mm)
L = 1e-3
x, C = solve_steady_slab(D, V, KM_ZERO_ORDER, C0, L, nx=400)
exact = C0 - V / (2 * D) * (2 * L * x - x ** 2)
err = np.max(np.abs(C - exact)) / C0
assert err < 1e-4, f"slab error {err:.2e}"

# Krogh cylinder
a, Rt = 5e-6, 1e-3
r, C = solve_krogh_mm_radial(D, V, KM_ZERO_ORDER, C0, a, Rt, nr=2000)
exact = C0 + V / (4 * D) * (r ** 2 - a ** 2) - V * Rt ** 2 / (2 * D) * np.log(r / a)
assert exact.min() > 0
err = np.max(np.abs(C - exact)) / C0
assert err < 2e-3, f"Krogh error {err:.2e}"

# Depleted slab: oxygen reaches zero at the zero-order critical depth
x, C = solve_steady_slab(D, V, KM_ZERO_ORDER, C0, 5e-3, nx=2000)
# Exact zero-order profile: C = V/(2D) (dcrit - x)^2 up to dcrit = sqrt(2 D C0 / V), then 0,
# so C falls below thr at x = dcrit - sqrt(2 D thr / V).
dcrit = np.sqrt(2 * D * C0 / V)
thr = 0.01
d = penetration_depth_from_threshold(x, C, thr)
d_exact = dcrit - np.sqrt(2 * D * thr / V)
assert abs(d - d_exact) / d_exact < 0.01, f"depth to {thr} mM: {d:.4e} vs exact {d_exact:.4e}"
assert C[x > dcrit * 1.01].max() < 1e-6, "oxygen should be ~0 beyond the critical depth"

# Michaelis-Menten (realistic Km) stays between 0 and C0 and decreases monotonically
x, C = solve_steady_slab(D, V, 1e-3, C0, 3e-3, nx=400)
assert C.min() >= 0 and C.max() <= C0 + 1e-12 and np.all(np.diff(C) <= 1e-12)

print(f"OK: slab and Krogh match the exact zero-order solutions; "
      f"depth to {thr} mM {d * 1e3:.3f} mm (exact {d_exact * 1e3:.3f} mm)")
