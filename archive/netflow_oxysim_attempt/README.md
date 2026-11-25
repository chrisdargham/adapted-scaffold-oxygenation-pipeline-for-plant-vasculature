# Archive: NetFlowV2 → oxygen-solver coupling attempt (May–June 2026)

These scripts tried to feed NetFlowV2 flows directly into an oxygen solver,
instead of the solver building its own flow estimate from the vein mask. **None
of them produced any result reported in this repository**, and none is called by
the pipeline, the demo or the tests. They are kept so that every piece of code
written for the project is in one place.

Paths in their docstrings (`scripts\...`, `MySim\...`, `results_oxy\...`) refer
to the original Windows working folder, not to this repository.

## Files

| File | What it was for |
|---|---|
| `oxy_sim.py` | Original image-based oxygen solver (reads a vein image, not a `.mat`). `src/solver/oxy_sim_mat.py` was derived from it. |
| `netflow_to_oxysim_bridge_final.py` | NetFlowV2 `NetworkNew.dat` → hemodynamic cache (`*_hemo.npz`) on a pixel grid, in the solver's array order. |
| `netflow_to_oxysim_bridge_fixed.py` | Variant that classifies inlets/outlets from flow sign. Lacks arrays the wrapper needs. |
| `netflow_to_oxysim_bridge_compat.py` | `_fixed` plus the arrays the wrapper needs. |
| `oxy_sim_netflow.py` | Wrapper: runs `oxy_sim.py` with a bridge cache instead of its own flow model. |
| `run_oxy_batch_final.py` | Batch runner: `.mat` → preprocessing → NetFlowV2 → bridge → wrapper, swept over `P_wall` and `Vmax`. |
| `analyze_oxy_results_final.py` | Aggregates the wrapper's `fields.npz` outputs into per-run and per-condition tables. |
| `sim_config_fixed_1200s.json` | One wrapper config (1,200 s run). |
| `graph_oxygen_netflow_solver.py` | Oxygen solved on the NetFlowV2 segment graph, then a distance-based tissue estimate. |
| `oxygen_distance_krogh_solver.py` | Tissue estimate from distance to the nearest perfused vessel only. |
| `keff_o2_concentration_screen.py` | Same, with an effective exchange coefficient lowering the source concentration. |
| `patch_oxy_zero_velocity.py` | Patch to `oxy_sim.py` to skip zero-speed vessel pixels (already applied in `oxy_sim.py` here). |
| `sanitize_hemo_for_oxysim.py` | Removes zero-flow pixels from a bridge cache. |
| `prepare_network_autoinlet.py`, `leafveincnn_to_netflow_autoinlet.py` | Variants of the `src/preprocessing/` scripts with `--auto-inlet` (inlet = terminal nearest the image edge) and an orphan-inlet fallback. Renamed here to avoid clashing with the maintained versions. `run_oxy_batch_final.py` expects these, passed as `--prepare` / `--exporter`. |

`patch_netflow_boundary_pressures.py` from the same period *was* used (it made the
May NetFlowV2 input) and is in `src/preprocessing/`.

## Known problems (why this was not used)

Checked in November 2026 on the May leaf network. Fix these before reusing any of it.

- `_fixed` and `_compat` store arrays as (y, x); `oxy_sim.py` expects (x, y). On a square grid the shape check passes and the leaf is silently transposed.
- `_compat`'s `lift_ix`/`lift_iy` are a transpose, not an identity map: 84% of perfused vessel pixels read lumen oxygen from a non-skeleton pixel (0).
- At 384 × 384 (33 × 50 µm pixels) the bridges mark 52–75% of the domain as vessel.
- "Perfused" counts any segment with nonzero flow (`_compat`: 76%) or any raster-connected pixel (`_final`: 100%); `docs/errata.md` point 5 shows 0.5% of vein length carries at least 0.1% of the inflow.
- `oxy_sim.py`'s lumen transport ignores flow direction and outlets and floors every speed at `u_min`; the graph solver on the same network depletes vessel oxygen within the first segments. The two disagree completely.
- `run_oxy_batch_final.py` defaults to `T = 0.5 s` (not steady state), reuses NetFlowV2's `Current/` folder without clearing it, and its `--auto-inlet` plus boundary outlets reproduce the May short-circuit.
- `oxy_sim.py` ignores `--config` (always reads `sim_config.json`), crashes on `polarity: "auto"`, and still has the explicit exchange update that overshoots at high `P_wall` (`docs/errata.md` point 6).
- `analyze_oxy_results_final.py` averages `..._fixed` runs together with the originals of the same condition.
