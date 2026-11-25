#!/usr/bin/env python3
"""
patch_netflow_boundary_pressures.py

Patch only the boundary pressure section of a NetFlowV2 Network.dat file.
It does not touch segment lines.

This is the step that produced the May NetFlowV2 input kept in
data/leaf_ACJ01/netflow_may/Network.dat: inlet node 1836 at 40 mmHg and the 44
outlets at 39.9955 mmHg (pressure drop 0.0045 mmHg). See
configs/leaf_ACJ01_may_preprocessing.json, and docs/errata.md point 5 for why
that pressure drop and outlet placement perfused almost none of the leaf.

Example (the May command; run from the repository root):
  python src/preprocessing/patch_netflow_boundary_pressures.py \
    --input  <prepare_network.py output>/network_ready.dat \
    --output data/leaf_ACJ01/netflow_may/Network.dat \
    --inlet-node 1836 --pin 40.0 --pout 39.9955

Check that it reproduces the kept file:  python tests/test_patch_boundary_pressures.py
"""

from __future__ import annotations
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--inlet-node", required=True, type=int)
    ap.add_argument("--pin", type=float, default=40.0)
    ap.add_argument("--pout", type=float, default=39.9955)
    args = ap.parse_args()

    lines = args.input.read_text(errors="ignore").splitlines()

    idx = None
    for i, line in enumerate(lines):
        if "boundary nodes" in line.lower():
            idx = i
            break
    if idx is None:
        raise RuntimeError("Could not find boundary node section.")

    nbc = int(float(lines[idx].split()[0]))

    # Boundary header is usually idx+1; data start idx+2.
    start = idx + 2
    changed = 0
    inlet_seen = False
    for j in range(start, start + nbc):
        parts = lines[j].split()
        if len(parts) < 4:
            continue
        node = int(parts[0])
        bctype = int(parts[1])

        # Only pressure boundary conditions are patched.
        if bctype != 0:
            continue

        if node == args.inlet_node:
            parts[2] = f"{args.pin:.6f}"
            inlet_seen = True
        else:
            parts[2] = f"{args.pout:.6f}"
        lines[j] = " ".join(parts)
        changed += 1

    if not inlet_seen:
        raise RuntimeError(f"Inlet node {args.inlet_node} not found in boundary section.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {args.output}")
    print(f"[OK] patched {changed} pressure boundary rows")
    print(f"[OK] inlet node {args.inlet_node}: {args.pin:.6f} mmHg")
    print(f"[OK] outlet nodes: {args.pout:.6f} mmHg")
    print(f"[OK] deltaP: {args.pin - args.pout:.6f} mmHg")

if __name__ == "__main__":
    main()
