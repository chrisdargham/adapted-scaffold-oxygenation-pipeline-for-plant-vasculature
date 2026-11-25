"""patch_netflow_boundary_pressures.py must rewrite only boundary pressures, and
reproduce the kept May input exactly. Run from the repository root."""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "preprocessing" / "patch_netflow_boundary_pressures.py"
MAY = ROOT / "data" / "leaf_ACJ01" / "netflow_may" / "Network.dat"


def patch(src, dst, pin, pout):
    subprocess.run([sys.executable, str(SCRIPT), "--input", str(src), "--output", str(dst),
                    "--inlet-node", "1836", "--pin", str(pin), "--pout", str(pout)],
                   check=True, capture_output=True)


with tempfile.TemporaryDirectory() as tmp:
    a, b = Path(tmp) / "a.dat", Path(tmp) / "b.dat"
    patch(MAY, a, 40.0, 35.0)                       # change the outlets ...
    may_lines, a_lines = MAY.read_text().splitlines(), a.read_text().splitlines()
    changed = [i for i, (x, y) in enumerate(zip(may_lines, a_lines)) if x != y]
    assert len(may_lines) == len(a_lines) and len(changed) == 44, len(changed)
    assert all("35.000000" in a_lines[i] for i in changed)
    patch(a, b, 40.0, 39.9955)                      # ... and put them back
    # The May file was written on Windows (CRLF line endings); compare the content.
    assert b.read_text().splitlines() == may_lines, "does not reproduce the May input"

print("OK: only the 44 outlet pressure rows change, and 40 / 39.9955 mmHg reproduces the May input exactly (ignoring CRLF line endings)")
