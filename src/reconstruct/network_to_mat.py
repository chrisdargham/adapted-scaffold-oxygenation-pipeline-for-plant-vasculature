"""Rebuild a LeafVeinCNN-style .mat file from a NetFlowV2 network file.

The original LeafVeinCNN export of the leaf was lost, but the prepared
NetFlowV2 network survived. That file has the coordinates (um) and diameter of
every segment. This script draws each segment back onto a pixel grid at the
original calibration (9.62 um/px), so the oxygen solver can run on the real
leaf again.

What is kept: vein positions, connectivity and diameters as prepared for
NetFlowV2 (including the 150 um diameter cap).
What is lost: the exact vein outline from the CNN; veins become straight
segments of constant width between nodes.

Run from the repository root:

    python src/reconstruct/network_to_mat.py \
        --network data/leaf_ACJ01/Network.dat \
        --out data/leaf_ACJ01/ACJ01_rebuilt.mat
"""
import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from skimage.draw import line
from skimage.morphology import dilation, disk


def read_network(path):
    """Return segments (start, end, diameter) and node coordinates from Network.dat."""
    lines = Path(path).read_text().splitlines()
    box_x, box_y = (float(v) for v in lines[1].split()[:2])
    nseg = int(lines[6].split()[0])
    segs = []
    for ln in lines[8:8 + nseg]:
        p = ln.split()
        segs.append((int(p[2]), int(p[3]), float(p[4])))
    i = 8 + nseg
    nnod = int(lines[i].split()[0])
    xy = {}
    for ln in lines[i + 2:i + 2 + nnod]:
        p = ln.split()
        xy[int(p[0])] = (float(p[1]), float(p[2]))
    return segs, xy, box_x, box_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--um-per-px", type=float, default=9.62)
    args = ap.parse_args()

    segs, xy, box_x, box_y = read_network(args.network)
    mpp = args.um_per_px
    ncols = int(round(box_x / mpp)) + 1
    nrows = int(round(box_y / mpp)) + 1

    def to_rc(node):
        x, y = xy[node]
        return nrows - 1 - int(round(y / mpp)), int(round(x / mpp))   # y axis points up

    # Draw every centreline, labelled with its radius in pixels.
    radius_px = np.zeros((nrows, ncols), dtype=np.int16)
    width_um = np.zeros((nrows, ncols), dtype=np.float32)
    skeleton = np.zeros((nrows, ncols), dtype=bool)
    for a, b, d in segs:
        (r0, c0), (r1, c1) = to_rc(a), to_rc(b)
        rr, cc = line(r0, c0, r1, c1)
        skeleton[rr, cc] = True
        r = max(1, int(round(d / mpp / 2)))
        radius_px[rr, cc] = np.maximum(radius_px[rr, cc], r)
        width_um[rr, cc] = np.maximum(width_um[rr, cc], d)

    # Thicken each centreline by its own radius.
    bw = np.zeros_like(skeleton)
    for r in np.unique(radius_px[radius_px > 0]):
        bw |= dilation(radius_px == r, disk(int(r)))

    # Spread the diameter from the centreline to the whole vein width (nearest centreline pixel).
    from scipy.ndimage import distance_transform_edt
    _, (ir, ic) = distance_transform_edt(~skeleton, return_indices=True)
    width_full = np.where(bw, width_um[ir, ic], 0).astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(args.out, {
        "bwFullWidth": bw.astype(np.uint8),
        "skeleton": skeleton.astype(np.uint8),
        "MicronPerPixel": mpp,
        "widthCalibrated": width_full,          # um
    }, do_compression=True)
    print(f"Saved {args.out}: {nrows} x {ncols} px, {len(segs)} segments, "
          f"vein area {100 * bw.mean():.1f}% of image")


if __name__ == "__main__":
    main()
