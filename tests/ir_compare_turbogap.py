#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""Head-to-head between MADLIB's IR pipeline and TurboGAP's ir_fft.f90.

TurboGAP's is a translation of GPUMD's TNEP/spectroscopy.py and is verified
against it to 3.5e-14 over eleven configurations, so agreeing with it to
machine precision is how this module inherits an external reference. An
estimator written from the definition alone would have none.

    ir_compare_turbogap.py <madlib driver> <turbogap driver> <dipoles>
"""
import subprocess
import sys

import numpy as np

MAD, REF, DIP = sys.argv[1], sys.argv[2], sys.argv[3]

cfgs = []
for win in ("hann","bartlett","welch","lorch","none","blackman"):
    cfgs.append((1.0, win, 0.1, 4000.0, 10, "gaussian", "harmonic", 300.0, 100.0))
for qc in ("harmonic","classical","quadratic","linear","none"):
    cfgs.append((1.0, "hann", 0.1, 4000.0, 10, "gaussian", qc, 300.0, 100.0))
for sk,kind in ((0,"gaussian"),(1,"gaussian"),(5,"gaussian"),(25,"gaussian"),(10,"box"),(3,"box")):
    cfgs.append((1.0, "hann", 0.1, 4000.0, sk, kind, "harmonic", 300.0, 100.0))
for r in (0.05, 0.2, 0.5, 1.0):
    cfgs.append((1.0, "hann", r, 4000.0, 10, "gaussian", "harmonic", 300.0, 100.0))
for dt in (0.25, 0.5, 2.0):
    cfgs.append((dt, "hann", 0.1, 4000.0, 10, "gaussian", "harmonic", 300.0, 100.0))
for nm in (500.0, 1500.0, 16000.0):
    cfgs.append((1.0, "hann", 0.1, nm, 10, "gaussian", "harmonic", 300.0, 100.0))
for T in (50.0, 1000.0):
    cfgs.append((1.0, "hann", 0.1, 4000.0, 10, "gaussian", "harmonic", T, 100.0))

worst, worstcfg, nbad = 0.0, None, 0
for c in cfgs:
    a = [str(v) for v in c]
    r1 = subprocess.run([REF,"spectrum",DIP]+a+["/tmp/mad_ir_ref.dat"], capture_output=True, text=True)
    r2 = subprocess.run([MAD,DIP]+a+["/tmp/mad_ir_got.dat"], capture_output=True, text=True)
    if r1.returncode or r2.returncode:
        if r1.returncode and r2.returncode:
            continue
        print(f"    MISMATCH in acceptance {c}: turbogap={r1.returncode} madlib={r2.returncode}"); nbad+=1; continue
    A = np.loadtxt("/tmp/mad_ir_ref.dat"); B = np.loadtxt("/tmp/mad_ir_got.dat")
    if A.shape != B.shape:
        print(f"    SHAPE {c}: {A.shape} vs {B.shape}"); nbad+=1; continue
    scale = np.abs(A).max(axis=0); scale[scale==0]=1.0
    d = (np.abs(A-B)/scale).max()
    if d > worst: worst, worstcfg = d, c
    if d > 1e-12: print(f"    {d:.2e}  {c}"); nbad+=1
print(f"    {len(cfgs)} configurations, worst relative difference {worst:.2e}")
print(f"    worst at window={worstcfg[1]} qc={worstcfg[6]} ratio={worstcfg[2]} smooth={worstcfg[4]}{worstcfg[5]}")
sys.exit(1 if nbad else 0)
