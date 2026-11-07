#!/bin/bash
# Regenerate nmr_aniso_target.dat: the synthetic 'experiment' the
# fix mad/nmr/aniso tests are biased towards. Two Gaussian lines on the same
# axis the test decks use, so the fit has somewhere to go and the gradient is
# not evaluated at a minimum.
set -euo pipefail

OUT=$(dirname "$0")/nmr_aniso_target.dat
XMIN=75.0
XMAX=95.0
NPTS=101
CENTRES="82.0 88.0"
HEIGHTS="1.0 0.6"
WIDTH=1.2

write_target() {
    python3 - "$OUT" "$XMIN" "$XMAX" "$NPTS" "$WIDTH" "$CENTRES" "$HEIGHTS" <<'PY'
import sys
out, xmin, xmax, npts, width, centres, heights = sys.argv[1:8]
xmin, xmax, npts, width = float(xmin), float(xmax), int(npts), float(width)
centres = [float(c) for c in centres.split()]
heights = [float(h) for h in heights.split()]
with open(out, "w") as fh:
    fh.write("# shift/ppm  intensity  (synthetic two-line target, "
             "make_nmr_aniso_target.sh)\n")
    for k in range(npts):
        x = xmin + (xmax - xmin) * k / (npts - 1)
        y = sum(h * 2.718281828459045 ** (-0.5 * ((x - c) / width) ** 2)
                for c, h in zip(centres, heights))
        fh.write(f"{x:.6f} {y:.12e}\n")
PY
}

write_target
head -2 "$OUT"
