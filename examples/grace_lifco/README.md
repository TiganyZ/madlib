# Fine-tuning GRACE on MAD configurations: Li–F–C–O

A foundation model is weakest where it has not been, and MAD sends a simulation
exactly there: towards a structure an experiment says exists, by paths the
model's own dynamics would not take. This example uses that. It runs
`GRACE-FS-OMAT` under an X-ray bias, labels a diverse selection of the frames
the bias produced, fine-tunes the model on them, and runs MAD again with the
fine-tuned model reporting its own extrapolation grade.

The system is an amorphous LiF/Li₂CO₃ mixture (48 LiF + 16 Li₂CO₃, 192 atoms),
the inorganic part of a battery SEI.

```
01 structures      random LiF + Li2CO3 packing; strained/rattled crystals of both
02 foundation      GRACE-FS-OMAT exported for pair_style grace/fs
03 melt_quench     NPT 1300 K -> 300 K with the foundation model
04 reference       the "experiment": XRD of labeller MD at 300 K
05 mad             MAD ladder (3 temperatures x 3 bias strengths, + unbiased), 30 ps each, a frame per ps
06 select          farthest-point selection in GRACE feature space, 70 frames per temperature
07 label           labeller model (default) or VASP via autoplex_soap_turbo
08 finetune        gracemaker, full FS finetune; export + active set
09 validate        foundation vs finetuned against the labels
10 mad_finetuned   the same bias with the finetuned model and its gamma
```

## Run it

On alt (any machine: add `machines/<name>.sh` and set `MACHINE`):

```sh
bash setup/make_venv.sh        # uv venv from the hashed lock
bash setup/fetch_models.sh     # checked against setup/models.sha256
bash setup/build_lammps.sh     # pinned LAMMPS + GRACE ML-PACE + MADLIB
./run_all.sh                   # every stage; a finished stage is skipped
```

Outputs land in `$PREFIX/runs/$NAME/<stage>/` (`/l/zarrout1/madlib-grace/runs/lifco/` on alt).
To rerun a stage, delete its directory.

## Reproducible

Everything that decides a number is pinned in `env.sh`, `params.sh` or a lock:

| | pinned by |
|---|---|
| LAMMPS | upstream commit `LAMMPS_COMMIT` |
| GRACE pair styles | `yury-lysogorskiy/lammps` commit `GRACE_COMMIT`, its `ML-PACE` overlaid on upstream |
| MADLIB | the commit of this checkout, written to `build/VERSIONS` |
| Python | `setup/requirements.txt`, hashed, installed with `--require-hashes` |
| models | `setup/models.sha256`; `fetch_models.sh` refuses a changed download |
| randomness | `SEED` in `env.sh` drives packing, velocities, Langevin, FPS, split and `gracemaker` |
| parallel decomposition | `NRANKS` in `params.sh`: LAMMPS is bitwise reproducible only at a fixed rank count |

TensorFlow runs on the CPU (`CUDA_VISIBLE_DEVICES=`), and the fine-tune sets
`TF_DETERMINISTIC_OPS=1`. `check_reproducible.sh a b` compares two complete runs
(`NAME=a ./run_all.sh; NAME=b ./run_all.sh`) stage by stage. Two runs on alt
(2026-10-22) agreed **bitwise** at every stage: structures, quench, target,
every MAD trajectory, selection, labels, the fine-tuned `saved_model.yaml` and
`.asi`, and the final MAD summary. VASP labels are reproducible only to the
extent VASP is; `LABEL=vasp` runs are checked from stage 07 on numerically.

## Labels

**`LABEL=grace`** (default) labels with `GRACE-2L-OMAT-large-ft-AM-fp64`, a
stand-in for DFT so the whole pipeline runs on one workstation in a few hours.
The fine-tuned FS model then learns the larger model's surface where MAD took
the foundation model, which is the demonstration; it is not DFT accuracy.

**`LABEL=vasp`** is the real thing: VASP statics on Roihu, through
`autoplex_soap_turbo/workflows/lifco_mad` and its jobflow-remote project
(`AUTOPLEX_ENV`, `JF_PROJECT` in `machines/alt.sh`). `stages/vasp_label.sh`
writes a per-run settings file (flow `lifco_mad_<NAME>`, frames from
`06_select/to_label.xyz`), submits once, polls `status.py` every `VASP_POLL`
seconds until every static has finished, and harvests them by name. A frame
whose static failed is dropped from the training set.

The level of theory is OMat24's, the data GRACE-FS-OMAT was trained on:
pymatgen's `MPRelaxSet` with PBE.54 POTCARs and `ALGO Normal`
([arXiv:2410.12771](https://arxiv.org/abs/2410.12771)) — PBE, 520 eV, Γ only at
MP's k-point density for these cells. `ISPIN 1` (the system is closed shell) and
`EDIFF 1e-6` go on top; neither changes the energy surface.

```sh
NAME=lifco VASP_PILOT=4 bash stages/vasp_label.sh   # cost 4 frames first, separate flow
NAME=lifco_vasp LABEL=vasp ./run_all.sh               # everything, VASP labels
```

Stages 01–06 are deterministic, so `cp -r runs/lifco/0[1-6]_* runs/lifco_vasp/`
skips straight to labelling. Budget: 210 statics, 200 of them 192-atom cells
at a 2×2×2 mesh on one 128-rank node each. Run the pilot before the batch.

## Fine-tuning on a Triton GPU

```sh
FT_WHERE=triton ./run_all.sh
```

Stage 08 then writes the same `input.yaml`, submits it as a jobflow-remote job
(`autoplex_soap_turbo/workflows/grace_finetune`, one A100), waits, and unpacks
the model into `08_finetune/seed/$SEED`; stages 09–10 are unchanged. The GRACE
venv on Triton is built from the same lock: `MACHINE=triton bash
setup/make_venv.sh && MACHINE=triton bash setup/fetch_models.sh`.

Measured: 10 min on the A100 against 55 min on alt's CPU (12 against ~990
µs/atom per evaluation), and the model's test errors agree with the CPU fit to
the fourth printed digit. It is **not** bitwise reproducible:
`TF_DETERMINISTIC_OPS` breaks GRACE under XLA on GPU, so the GPU fit runs without
it, and MD driven by the two models diverges after a few ps as any two MD runs do.
alt's RTX 5070 cannot do this: TF 2.20 has no sm_120 kernels.

## What it gives

Measured on alt (`SEED=20261022`, 8 ranks, TensorFlow on 10 CPU cores):

| stage | result | wall |
|---|---|---|
| 03 | density 1.89 g/cm³ at 300 K | 7 min |
| 04 | target = mean XRD of 21 labeller frames | 47 min |
| 05 | unbiased `rel_error` 0.16 / 0.30 / 0.41 at 300 / 600 / 900 K; at 300 K the bias takes it to 0.061 / 0.035 / 0.021 (0.005 / 0.02 / 0.05 eV/atom) | 28 min |
| 06–07 | 200 of 480 frames; 170 train (with 10 crystals), 40 test | 15 min |
| 08 | 50 BFGS iterations | 55 min |

Held-out MAD frames, against the labeller (stage 09):

| | E MAE (meV/atom) | F MAE (meV/Å) | stress MAE (GPa) |
|---|---|---|---|
| GRACE-FS-OMAT | 11.1 | 239 | 0.66 |
| fine-tuned | 1.3 | 77 | 0.075 |

MAD at 300 K, same start, seed and gamma (stage 10):

| | `rel_error` | D | max γ |
|---|---|---|---|
| foundation, unbiased | 0.141 | 1.23 | |
| fine-tuned, unbiased | 0.099 | 0.61 | 0.84 |
| foundation, gamma 12.5 | 0.021 | 0.025 | |
| fine-tuned, gamma 12.5 | 0.021 | 0.027 | 0.82 |

Unbiased, the fine-tuned model's own liquid sits half as far from the target as
the foundation model's: it has learned the structure the bias had to impose.
Under the bias both match equally well, and the fine-tuned model stays inside
its training data while doing so (γ < 1). The fit had not converged at 50
iterations (test force MAE still falling); `FT_ITERS=100` is cheap to try.

### With VASP labels (`lifco_v2`)

`NAME=lifco_v2 LABEL=vasp FT_WHERE=triton FT_ITERS=200 ./run_all.sh`, on
2026-10-23: 30 ps MAD runs, 70 frames per temperature ≥ 1 ps apart, 220 VASP
statics at OMat24's level of theory on Roihu (~2 min each, ~30 min for the
batch), 200 BFGS iterations on one Triton A100 (30 min). Errors against VASP on
the 42 held-out frames:

| | E MAE (meV/atom) | F MAE (meV/Å) | stress MAE (GPa) |
|---|---|---|---|
| GRACE-FS-OMAT | 17.9 (4.7 after per-element shift) | 229 | 0.37 |
| fine-tuned | 0.47 | 44 | 0.044 |

Force MAE by temperature, fine-tuned: 34 / 42 / 55 meV/Å at 300 / 600 / 900 K
(foundation 190 / 227 / 269). Unbiased at 300 K the fine-tuned model's structure
is again about half as far from the target (D 1.46 → 0.77); under gamma 11.4
both reach `rel_error` 0.022, the fine-tuned model with max γ 1.08 — just past
its active set, so those frames are the next ones worth labelling.

## Using the fine-tuned model with MADLIB

```lammps
pair_style  grace/fs extrapolation
pair_coeff  * * saved_model.yaml saved_model.asi Li C O F
fix         gam all pair 500 grace/fs gamma 1          # per-atom extrapolation grade

compute     gr  all mad/pdf r 0.1 6.0 201 kde 0.1
fix         xrd all mad/xrd c_gr target target_xrd.dat q 1.0 10.0 181 &
                output qFq species Li C O F gamma 12.5
```

`comm_modify cutoff 9.0`: the PDF reaches `rmax + trunc*kde = 6.8` Å, past
GRACE's 6 Å ghost shell. `lammps/in.mad` is the complete input.

## Adapting it

Change `SPECIES` and the packing in `stages/structures.py`, the element lists in
`lammps/in.*` (type order must match), the target in stage 04 (a measured
pattern replaces `target_xrd.dat` as two columns, `q  qF(q)`, covering the fix's
q range), and the knobs in `params.sh`. `MAD_EBIAS` is the bias energy per atom
at the starting dissimilarity, so it carries across systems where a raw gamma
does not.
