# Driving a MACE foundation model with MADlib

The smallest useful MADlib run with a machine-learned potential: take a
foundation model off the shelf, add a measured diffraction pattern, and let
the two decide the structure together.

Nothing here is specific to the model or the system. `pair_style
symmetrix/mace` supplies the energy, `compute mad/pdf` and `fix mad/xrd`
supply the experiment, and swapping either is a one-line change.

## What you need

A LAMMPS with the MADLIB package and `pair_style symmetrix/mace`:

```sh
tools/install_turbogap_lammps.sh          # stages: deps python lammps symmetrix madlib build
```

A foundation model in the format symmetrix reads, and a target pattern as two
columns, `q` and the quantity named by `output`.

## Run it

```sh
lmp -in lammps/in.mad -var model mace-mp-0.json -var species C \
    -var target your_pattern.fq -var gamma 10.0 -var temp 1000
```

## The three lines that matter

```
pair_style      symmetrix/mace
pair_coeff      * * ${model} ${species}

compute gr  all mad/pdf r 0.1 6.0 151 kde 0.1
fix     xrd all mad/xrd c_gr target ${target} q 1.0 10.0 181 output qFq &
            species ${species} gamma ${gamma}
```

`compute mad/pdf` builds the smeared pair distribution, `fix mad/xrd`
transforms it, compares it with the file, and adds the gradient of the
mismatch to the forces. `gamma` is the only knob: 0 is the plain potential,
large is the pattern overruling the physics. There is no useful default --
scan it, and read the front rather than one point.

## Reading the output

`f_xrd` is the bias energy, `f_xrd[2]` the dimensionless relative error,
`f_xrd[3]` the raw dissimilarity. Watch `f_xrd[2]`: if it is flat, gamma is
too small to matter; if the temperature climbs and the energy with it, gamma
is too large.

**Report `c_pepair`, not `pe`.** `fix mad/*` sets `thermo_energy` on by
default, so `pe` contains the bias. An energy axis with the bias in it is not
the potential's energy and cannot be compared against anything.

## Two things worth checking before trusting a number

**r_max.** The pair distribution is cut at `r 0.1 6.0`, which caps the
q-resolution at about 2*pi/6. If the measured pattern has features narrower
than that, the prediction cannot match them at any gamma and the misfit has a
floor that is the grid, not the structure. Scan r_max on a fixed structure and
take the smallest value whose misfit has stopped moving -- and do it against
the target in hand, because the answer does not transfer between experiments.

**r_max < L/2.** Beyond half the box the pair distribution counts periodic
images of the cell as if they were neighbours. `compute mad/pdf` needs a ghost
cutoff of `r_max + 8*kde + skin`, so the cell has to be big enough for both.
