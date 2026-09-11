# TurboGAP and LAMMPS, side by side

Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)

What the two codes call things, where the data actually lives, and which
differences forced a change in MADLIB rather than a transcription. Written
while porting TurboGAP's experimental-observables machinery; every claim here
is something the port ran into.

---

## 1. Where the atoms are

**TurboGAP.** Loose module-level arrays in the driver, `turbogap.f90`, passed
down through long argument lists.

| quantity | array | shape |
|---|---|---|
| positions | `positions` | `(3, n_sites)` |
| velocities | `velocities` | `(3, n_sites)` |
| forces | `forces` | `(3, n_sites)` |
| species | `xyz_species`, `species` | `(n_sites)` |
| cell | `a_box`, `b_box`, `c_box` | three `(3)` vectors |

Fortran is column-major, so `positions(1:3, i)` is one atom, contiguous. There
is a `type(image)` in `types.f90` holding copies of all of it, but it is a
*snapshot*: Monte Carlo and nested sampling copy out to an image and copy back,
and the driver's loose variables remain the truth. TurboGAP's own `TODO.md` is
blunt about this being half-finished.

**LAMMPS.** One `Atom` object, reached as `atom->` from anything.

| quantity | member | shape |
|---|---|---|
| positions | `atom->x` | `double **`, `[nlocal+nghost][3]` |
| velocities | `atom->v` | `double **`, same |
| forces | `atom->f` | `double **`, same |
| type | `atom->type` | `int *`, 1-based into `ntypes` |
| charge | `atom->q` | `double *`, only if the atom style has it |
| global ID | `atom->tag` | `tagint *` |
| molecule | `atom->molecule` | `tagint *`, only for molecular styles |
| group mask | `atom->mask` | `int *`, bitmask |
| cell | `domain->` | `boxlo/boxhi`, `xprd/yprd/zprd`, `xy/xz/yz` |

C is row-major, so `x[i][0..2]` is one atom, contiguous. The index ordering is
therefore *transposed* between the two codes: `positions(k, i)` against
`x[i][k]`.

The consequential difference is not the layout. It is that **`nlocal` is not the
number of atoms.** Each rank owns `nlocal` atoms and additionally holds
`nghost` copies of atoms owned elsewhere, which are real entries in the same
arrays, at indices `nlocal .. nlocal+nghost-1`. TurboGAP has nothing equivalent:
every rank holds every atom and the parallelism is purely a loop split.

Consequences MADLIB had to handle:

- A sum over atoms is a sum over `i < nlocal` followed by `MPI_Allreduce`.
  Including ghosts double-counts.
- A force *written* to a ghost is a real contribution that has to be
  communicated back. See section 5.
- Anything indexed per-atom must survive atoms migrating between ranks, so it
  is either recomputed each step or registered with `Atom` for migration.
- `atom->natoms` is the global count, a `bigint`, and is the one to use for a
  density or a concentration.

## 2. The neighbour list

**TurboGAP.** Three flat arrays built by `build_neighbors_list`
(`neighbors.f90:243`), with the pair data already in spherical coordinates:

```
n_neigh(1:n_sites)          neighbours of site i, INCLUDING ITSELF
neighbors_list(1:n_pairs)   flat, site by site
rjs, thetas, phis(1:n_pairs)  distance and direction of each pair
xyz(1:3, 1:n_pairs)         the separation vector as well
```

Pairs are flattened in site order, and **the first pair of each site is the
site itself, at `rjs = 0`**. That self pair is not padding: it is where
`dq_i/dr_i` lands in every derivative array, and dropping it silently loses the
central-atom term.

Periodicity is handled by building an explicit **supercell** large enough that
its faces are `2*rcut` apart (`neighbors.f90:169`), then folding neighbour
indices back with `mod(j - 1, n_sites) + 1`. Images are literal extra atoms.

**LAMMPS.** A `NeighList` object, requested rather than built:

```cpp
auto *req = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
req->set_cutoff(rcut + neighbor->skin);
...
neighbor->build_one(list);          // occasional lists only

list->inum                          // atoms with a list
list->ilist[ii]                     // -> local atom index i
list->numneigh[i]                   // neighbours of i, NOT including itself
list->firstneigh[i][jj] & NEIGHMASK // -> a neighbour index, local or ghost
```

Four differences that matter:

1. **No self pair.** TurboGAP's `n_neigh(i)` counts the atom itself; LAMMPS's
   `numneigh[i]` does not. Converting one to the other is off by one per site,
   and the missing entry is exactly the one carrying `dq_i/dr_i`.
2. **Cartesian, not spherical.** LAMMPS gives you indices; you compute
   `delx = x[i][0] - x[j][0]` yourself. Anything wanting `(r, theta, phi)`  -- 
   soap_turbo does -- converts at the call site.
3. **`NEIGHMASK`.** The top bits of a neighbour index encode special-bond
   exclusions. Using the raw value as an index is a silent out-of-bounds.
4. **Periodicity is already in it.** A neighbour may be a ghost, whose position
   is the correctly-imaged copy, so there is no minimum-image convention to
   apply and no supercell to build. `j` may be `>= nlocal`.

Half against full is a choice in LAMMPS and not in TurboGAP. MADLIB uses full
lists throughout, for a reason specific to fixes -- section 5.

## 3. Parallel decomposition

**TurboGAP.** Every rank holds every atom. Work is split by index range:

```fortran
i_beg = 1 + rank*(n_sites/ntasks + 1)     ! turbogap.f90:1079
```

and the comment above it reads "This is some trivial MPI parallelization to
make sure the code works fine", with spatial decomposition marked TO DO. Global
arrays are reduced to rank 0 and broadcast back, so all ranks agree.

**LAMMPS.** Spatial domain decomposition. A rank owns a region of the box and
the atoms currently in it; atoms migrate as they move. Nothing is replicated
except ghosts.

For a *global* observable -- and every observable MADLIB computes is one, a sum
over all atoms or all pairs -- the pattern is the same in both codes and is worth
stating once:

1. each rank accumulates its own partial spectrum;
2. **`MPI_Allreduce`**, so every rank holds the identical global residual  -- 
   reduce-to-one is not enough, because every rank then needs that residual to
   build its own forces;
3. each rank builds forces only for what it owns;
4. terms belonging to the *cell* rather than to any atom -- the volume part of
   the virial -- are added by **one rank only**, since LAMMPS all-reduces the
   fix virial. Every rank adding it gives an `nprocs`-fold stress.

TurboGAP guards step 4 with `rank == 0` for the same reason.

## 4. Where the physics goes

| TurboGAP | LAMMPS |
|---|---|
| a branch in `turbogap.f90`'s main loop | a `Fix` or `Compute` class, registered by a macro in its header |
| `params%do_xrd` etc., read by `read_files.f90` | constructor arguments, parsed with `utils::numeric` and friends |
| add into `forces` before the integrator | `post_force()`, which runs after the pair style |
| `energies_exp` summed into the total | `compute_scalar()`, with `energy_global_flag` set |
| `virial` accumulated in the same routine | the fix's own `virial[6]`, all-reduced by LAMMPS |
| everything in one binary | optional package, `-D PKG_MADLIB=on` |

The gain from the LAMMPS side is that a bias becomes a first-class object that
composes: it appears in `thermo_style`, contributes to `compute pe`, and is
therefore seen by the Metropolis test of `fix gcmc` and `fix atom/swap` without
any Monte Carlo code being written. TurboGAP has to fold the experimental term
into its own energy array and its own MC driver.

## 5. Force ordering -- the trap

`Verlet::run()` reverse-communicates ghost forces at `verlet.cpp:341`, and calls
`modify->post_force()` at `:347`. **A force written to a ghost atom inside
`post_force()` is silently dropped.**

That single fact settles two design points in opposite directions:

- **Pair-based observables use a full list and write only to local `i`.** Under
  a full list the ordered pair `(i,j)` is visited on the rank owning `i` and
  `(j,i)` on the rank owning `j`, so every atom's force is already complete
  locally and no communication is needed. This is also why TurboGAP writes to
  one end of each pair, though for a different reason.
- **A per-atom property's adjoint cannot avoid it.** `dq_i/dr_j` reaches the
  neighbours of `i`, ghosts included. MADLIB accumulates into its own buffer
  over `nlocal+nghost` and calls `comm->reverse_comm(this)`.

And the sharp edge on that second one: **a serial run still has ghosts.**
Guarding the reverse communication on `nprocs > 1` leaves the *serial* force
wrong by a constant, because periodic images are ghosts too.

## 6. Units

| | TurboGAP | LAMMPS *metal* |
|---|---|---|
| energy | eV | eV |
| length | Angstrom | Angstrom |
| **time** | **fs** | **ps** |
| mass | eV fs^2 / A^2 | g/mol |

Energies and lengths carry over unchanged, which is why porting the static
observables needed no conversions at all. Time does not. The IR spectrum uses
`33356.40952 cm^-1` per inverse fs, so the stored-frame interval has to be
converted out of picoseconds; getting it wrong rescales the whole wavenumber
axis while producing a spectrum that looks entirely plausible.

## 7. Reading a potential

TurboGAP's `read_files.f90` is 6.7k lines and parses everything -- the run's
input deck and the `.gap` potential -- into one `input_parameters` type.

MADLIB reads only the part of a `.gap` file that a local-property model needs,
in about 250 lines of C++ (`mad_gap_file.cpp`). One detail is worth recording
because it is invisible until it fails: **the sparse-set paths inside a `.gap`
file are relative to the directory TurboGAP was *run* from**, not to the file.
`gap_files/CO.gap` contains `local_property_qs = 'gap_files/...'`. A LAMMPS run
has no reason to share that working directory, so the parser tries the path as
given, then beside the potential file, then flattened beside it.

## 8. What is faithfully the same

Deliberately, so that numbers can be compared:

- the observable definitions, the Ashcroft-Langreth partial structure factors,
  and the `xrd_output` conventions;
- the scattering tables, transcribed from `exp_utils.f90`;
- `soap_turbo` itself and `local_property_predict`, vendored and called through
  a C binding rather than reimplemented;
- the `ir_fft` pipeline, translated statement by statement.

Measured agreement: core-electron binding energies to `1.2e-9` relative, the IR
pipeline to `1.5e-13` over 29 configurations.

Where MADLIB deliberately differs, `compat turbogap` restores TurboGAP's
behaviour -- including its single-precision constants, which is why the X-ray
form factors match it to `4e-16` in that mode and differ in the eighth digit
outside it.

## 9. Summary table

| | TurboGAP | LAMMPS |
|---|---|---|
| atom data | loose `(3, n_sites)` arrays in the driver | `atom->x[i][3]`, one `Atom` object |
| indexing | column-major, `(k, i)` | row-major, `[i][k]` |
| atom count | every rank has all of them | `nlocal` owned + `nghost` copies |
| neighbour list | flat arrays, spherical, self pair first | `NeighList`, indices only, no self pair |
| periodicity | explicit supercell of images | ghost atoms |
| parallelism | index-range split, all-replicated | spatial domains, atoms migrate |
| new physics | a branch in the main loop | a `Fix` / `Compute` class |
| time unit | fs | ps (*metal*) |
| potential input | one 6.7k-line reader | per-style parsing with `utils::` |
