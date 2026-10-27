# Fine-tuning GRACE on MAD configurations

The example has five steps:

1. Run a GRACE foundation model under an X-ray bias (MAD).
2. Select diverse frames from those runs.
3. Label the selected frames.
4. Fine-tune the model on them with `gracemaker`.
5. Run MAD again with the fine-tuned model, which reports its own extrapolation
   grade.

The system is an amorphous LiF/Li₂CO₃ mixture. Change it in `params.sh` and
`stages/structures.py`.

```
01 structures      random packing; strained/rattled crystals
02 foundation      the foundation model exported for pair_style grace/fs
03 melt_quench     NPT melt and quench with the foundation model
04 reference       the "experiment": XRD of labeller MD
05 mad             MAD over temperatures x bias strengths, plus unbiased runs
06 select          farthest-point selection in GRACE feature space
07 label           a larger GRACE model (default) or VASP via autoplex_soap_turbo
08 finetune        gracemaker fine-tune; export and active set
09 validate        foundation vs fine-tuned against the labels
10 mad_finetuned   the same bias with the fine-tuned model
```

## Run it

```sh
bash setup/make_venv.sh        # Python environment from the hashed lock
bash setup/fetch_models.sh     # foundation models, checked against setup/models.sha256
bash setup/build_lammps.sh     # pinned LAMMPS + GRACE + MADLIB
./run_all.sh                   # every stage; finished stages are skipped
```

For a new machine, add `machines/<name>.sh` and set `MACHINE`. Outputs go to
`$PREFIX/runs/$NAME/<stage>/`. Delete a stage's directory to rerun it.

## Options

| | |
|---|---|
| `LABEL=grace` (default) | labels from a larger GRACE model, so everything runs on one workstation |
| `LABEL=vasp` | VASP statics through `autoplex_soap_turbo/workflows/lifco_mad`; `VASP_PILOT=N bash stages/vasp_label.sh` costs N frames first |
| `FT_WHERE=triton` | the fine-tune runs as a jobflow-remote job on a GPU, through `autoplex_soap_turbo/workflows/grace_finetune` |
| `FT_ITERS` | number of fine-tuning iterations |

## Reproducible

LAMMPS, GRACE, MADLIB, the Python packages and the models are all pinned (see
`env.sh`, `setup/requirements.txt` and `setup/models.sha256`). `SEED` drives
every random choice. With TensorFlow on the CPU and a fixed `NRANKS`, two runs
agree bitwise. To compare two runs stage by stage, run
`check_reproducible.sh a b`.

## Using the fine-tuned model

```lammps
pair_style  grace/fs extrapolation
pair_coeff  * * saved_model.yaml saved_model.asi Li C O F
fix         gam all pair 500 grace/fs gamma 1          # per-atom extrapolation grade

compute     gr  all mad/pdf r 0.1 6.0 201 kde 0.1
fix         xrd all mad/xrd c_gr target target_xrd.dat q 1.0 10.0 181 &
                output qFq species Li C O F gamma 12.5
comm_modify cutoff 9.0
```

`lammps/in.mad` is the complete input.

## Adapting it

To adapt it to another system, change:

- `SPECIES` and the packing in `stages/structures.py`.
- The element lists in `lammps/in.*`, keeping the type order.
- The target in stage 04. A measured pattern replaces `target_xrd.dat`, as two
  columns `q  qF(q)` that cover the fix's q range.
- The parameters in `params.sh`. `MAD_EBIAS` is the bias energy per atom at the
  starting dissimilarity, so it carries across systems where a raw `gamma` does
  not.
