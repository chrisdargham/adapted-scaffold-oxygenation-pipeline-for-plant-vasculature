#!/usr/bin/env python3
"""
krogh_validation.py
===================
Steady one-dimensional reference solutions used by `oxy_sim_mat.py` when the
config sets "verify": true.

Both problems use Michaelis-Menten consumption R(C) = Vmax * C / (Km + C):

  Slab:   D C''(x) = R(C),                    C(0) = C0,  C'(L)  = 0
  Krogh:  D (1/r) d/dr (r dC/dr) = R(C),      C(a) = C0,  C'(Rt) = 0
          (a = vessel radius, Rt = tissue cylinder radius)

They are solved by second-order finite differences and Newton iteration.
With Km -> 0 and C > 0 everywhere, consumption is zero order and the exact
solutions are

  Slab:   C(x) = C0 - Vmax/(2D) * (2 L x - x^2)
  Krogh:  C(r) = C0 + Vmax/(4D) * (r^2 - a^2) - Vmax Rt^2/(2D) * ln(r/a)

`tests/test_krogh_validation.py` checks the solvers against these.

Units: SI lengths (m), D in m^2/s, Vmax in mM/s, Km and C in mM.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded


def _mm(C, Vmax, Km):
    Cp = np.maximum(C, 0.0)
    R = Vmax * Cp / (Km + Cp + 1e-30)
    dR = np.where(C > 0, Vmax * Km / (Km + Cp + 1e-30) ** 2, Vmax / max(Km, 1e-30))
    return R, dR


def _newton(lower, diag, upper, C, C0, Vmax, Km, tol=1e-12, itmax=200):
    """Solve  A C - R(C) = b  where A is tridiagonal (lower/diag/upper) acting on
    unknowns C[1:], with C[0] = C0 fixed (its effect enters through lower[0]).

    Uses continuation in Km: small Km makes R(C) nearly a step at C = 0 and plain
    Newton stalls at the depletion front, so the problem is first solved with a
    large Km and Km is then reduced tenfold at a time, warm-starting each solve."""
    km_steps = [Km]
    while km_steps[-1] < C0:
        km_steps.append(km_steps[-1] * 10.0)
    for km in reversed(km_steps):
        C = _newton_fixed_km(lower, diag, upper, C, C0, Vmax, km, tol, itmax)
    return C


def _newton_fixed_km(lower, diag, upper, C, C0, Vmax, Km, tol, itmax):
    n = C.size - 1
    for _ in range(itmax):
        u = C[1:]
        R, dR = _mm(u, Vmax, Km)
        Au = diag * u
        Au[1:] += lower[1:] * u[:-1]
        Au[:-1] += upper[:-1] * u[1:]
        Au[0] += lower[0] * C0
        F = Au - R
        ab = np.zeros((3, n))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = diag - dR
        ab[2, :-1] = lower[1:]
        du = solve_banded((1, 1), ab, -F)
        u_new = np.maximum(u + du, 0.0)
        C[1:] = u_new
        if np.max(np.abs(u_new - u)) < tol * max(C0, 1e-30):
            break
    return C


def solve_steady_slab(D, Vmax, Km, C0, L, nx=400):
    """Steady slab: supply at x = 0, no flux at x = L. Returns (x, C)."""
    x = np.linspace(0.0, L, nx + 1)
    h = x[1] - x[0]
    n = nx
    k = D / h ** 2
    lower = np.full(n, k)
    upper = np.full(n, k)
    diag = np.full(n, -2.0 * k)
    lower[-1] = 2.0 * k          # mirror ghost node for C'(L) = 0
    upper[-1] = 0.0
    C = np.full(nx + 1, float(C0))
    C[0] = C0
    C = _newton(lower, diag, upper, C, C0, Vmax, Km)
    return x, C


def solve_krogh_mm_radial(D, Vmax, Km, C0, a, Rt, nr=400):
    """Steady Krogh cylinder: C = C0 at r = a, no flux at r = Rt. Returns (r, C)."""
    r = np.linspace(a, Rt, nr + 1)
    h = r[1] - r[0]
    n = nr
    ri = r[1:]
    rm = ri - 0.5 * h            # face radii r_{i-1/2}
    rp = ri + 0.5 * h            # face radii r_{i+1/2}
    lower = D * rm / (ri * h ** 2)
    upper = D * rp / (ri * h ** 2)
    diag = -(lower + upper)
    # Last node: half control volume, zero flux at r = Rt.
    vol = 0.5 * h * (Rt - 0.25 * h)
    lower[-1] = D * rm[-1] / (h * vol)
    diag[-1] = -lower[-1]
    upper[-1] = 0.0
    C = np.full(nr + 1, float(C0))
    C = _newton(lower, diag, upper, C, C0, Vmax, Km)
    return r, C


def penetration_depth_from_threshold(x, C, thr):
    """First position where C falls below thr (linear interpolation).
    Returns x[0] if C starts below thr, and x[-1] if C never falls below it."""
    x = np.asarray(x, dtype=float)
    C = np.asarray(C, dtype=float)
    below = np.nonzero(C < thr)[0]
    if below.size == 0:
        return float(x[-1])
    i = int(below[0])
    if i == 0:
        return float(x[0])
    x0, x1, c0, c1 = x[i - 1], x[i], C[i - 1], C[i]
    return float(x0 + (thr - c0) * (x1 - x0) / (c1 - c0))
