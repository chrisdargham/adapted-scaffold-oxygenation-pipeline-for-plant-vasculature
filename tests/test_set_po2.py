"""Check that run_pipeline.set_po2_in_network writes the inlet PO2 (bug fixed Oct 2026).

Run from the repository root:  python tests/test_set_po2.py
"""
import importlib.util
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rp", ROOT / "src" / "pipeline" / "run_pipeline.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

src = ROOT / "data" / "leaf_ACJ01" / "netflow_may" / "NetworkNew.dat"
with tempfile.TemporaryDirectory() as tmp:
    dst = Path(tmp) / "Network.dat"
    rp.set_po2_in_network(src, dst, inlet_press=40.0, inlet_po2=150.0, outlet_po2=40.0)
    lines = dst.read_text().splitlines()
    j = next(i for i, l in enumerate(lines) if "boundary nodes" in l)
    nbc = int(lines[j].split()[0])
    rows = [l.split() for l in lines[j + 2:j + 2 + nbc]]
    inlet = [r for r in rows if float(r[2]) == 40.0]
    assert len(inlet) == 1 and float(inlet[0][4]) == 150.0, inlet
    assert all(float(r[4]) == 40.0 for r in rows if float(r[2]) != 40.0)
    # Everything outside the boundary rows must be unchanged.
    original = src.read_text().splitlines()
    assert lines[:j + 2] == original[:j + 2]
print(f"OK: inlet PO2 = 150 mmHg, {nbc - 1} outlets at 40 mmHg, rest of file unchanged")
