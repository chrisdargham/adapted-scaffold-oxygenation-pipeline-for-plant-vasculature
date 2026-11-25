#!/usr/bin/env python3
r"""
patch_oxy_zero_velocity.py

Purpose:
  Patch MySim/oxy_sim.py so compute_lumen_with_hemo does not divide by zero
  when a vessel-mask pixel has u_loc == 0.

Biological logic:
  zero-flow pixels are not convective oxygen suppliers. They are skipped in
  lumen-transport calculation instead of being assigned an artificial velocity.

Usage:
  cd "C:\Users\admin\Documents\final capstone"
  python scripts\patch_oxy_zero_velocity.py --file MySim\oxy_sim.py
"""

from pathlib import Path
import argparse
import shutil

OLD = "lam = (k_loc * ds) / u_loc"

NEW = """# Skip stagnant/non-perfused vessel pixels before dividing by velocity.
        # Biological interpretation: zero-flow pixels are not convective oxygen suppliers.
        if (not np.isfinite(u_loc)) or (u_loc <= 1e-12):
            continue
        lam = (k_loc * ds) / u_loc"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, type=Path)
    args = ap.parse_args()

    p = args.file
    text = p.read_text(encoding="utf-8", errors="ignore")

    if NEW in text:
        print("[OK] Patch already present.")
        return

    if OLD not in text:
        raise RuntimeError("Could not find target line: lam = (k_loc * ds) / u_loc")

    backup = p.with_suffix(p.suffix + ".bak_before_zero_velocity_patch")
    shutil.copy2(p, backup)

    text2 = text.replace(OLD, NEW, 1)
    p.write_text(text2, encoding="utf-8")

    print(f"[OK] Patched {p}")
    print(f"[OK] Backup saved as {backup}")
    print("[OK] Zero-speed vessel pixels will be skipped before lam division.")

if __name__ == "__main__":
    main()
