# Adding an observable, with a MACE (or any other) model behind it

MADLIB is two layers, and which one your new quantity belongs in decides
whether you write no code, ten lines, or a fix.

```
   model  ->  per-atom property  ->  observable  ->  bias
              compute mad/property/*   fix mad/*     E = gamma D / 2
```

The **property** is one number (or vector, or tensor) per atom. The
**observable** turns a set of those into something an experiment measures, and
compares it with a measured curve. Between them there is exactly one contract:

```
f_j -= sum_i lambda_i  d q_i / d r_j
```

The observable hands back one weight per atom, `lambda_i = dE/dq_i`, and the
property backend returns the force of the weighted sum `sum_i lambda_i q_i`.
That is one backward pass for a torch model — never a Jacobian.

---

## Case 1: your observable is a broadened per-atom scalar

Then there is nothing to write. `fix mad/xps` (registered as `mad/spectrum`
too, since nothing in it is about photoemission) broadens any per-atom scalar
into a spectrum; `fix mad/nmr` is the same with a shielding-to-shift map.

Train a MACE model on your per-atom target as if it were an atomic energy —
that is how a scalar comes out of an energy-shaped architecture — and point
the bridge at it:

```lammps
compute cebe all mad/property/python madlib_models ncomp 1 cutoff 12.0 &
            config "backend=mace model=cebe.model types=C,O"
fix     xps  all mad/xps c_cebe target measured.dat range 280 300 201 &
            sigma 0.4 gamma 5.0
```

`output=` picks which of the model's outputs to read (`node_energy` by
default, also `charges`, `atomic_dipoles`, `node_feats`), and `quantity=`
reduces it. A tensor-valued model can serve a scalar observable this way:
`quantity=iso` for a trace, `span`, `vzz`, `etaq`, or `component:zz`.

Which nucleus, element or site a spectrum is of is the **fix's group**, not a
property of the model: two groups over one compute are two spectra.

## Case 2: the property exists but no backend covers your model

Write the forward pass as one function and let the engine do the rest:

```python
# my_property.py
import torch

def charge_transfer(graph):
    """Whatever you like, as long as it is a torch expression of graph.pos."""
    d = graph.pos[graph.receiver] - graph.pos[graph.sender] + graph.shifts
    r = torch.linalg.norm(d, dim=1)
    w = torch.exp(-r / 2.0)
    q = torch.zeros(graph.pos.shape[0], dtype=r.dtype, device=r.device)
    q.index_add_(0, graph.receiver, w)
    return q
```

```lammps
compute q all mad/property/python madlib_models ncomp 1 cutoff 6.0 &
          config "backend=callable fn=my_property:charge_transfer types=C,O"
```

`graph` carries `pos` (on the autograd tape), `cell`, `numbers`, `sender`,
`receiver`, `shifts` and `nlocal`; see `python/madlib_models/geometry.py`. The
caching, the reduction, the backward pass and the periodic virial are already
handled. Return a dict for several quantities from one pass, and name one with
`output=`.

A whole model family — a framework with its own data pipeline — is a backend
module instead: four names, `KEYS`, `DEFAULT_OUTPUT`, `DEFAULT_QUANTITY`,
`identity(cfg)` and `build(cfg)`, returning an object with `forward(graph)`.
`backends/mace.py` is 130 lines including its warnings; copy it.

**The one thing to get right** is the edge direction. MACE and NequIP both
build their lists with ASE's `ijS` convention and aggregate onto `j`, so the
centre of an environment is `edge_index[1]` and the edge vector points from
neighbour to centre; LAMMPS pairs them the other way round. Backwards, every
vector flips, and a model with odd-parity features returns a plausible
different number. Test it against the framework's own pipeline — that is what
`tests/mace_reference.py` does, to 4e-16.

## Case 3: your observable is genuinely new

A new spectrum is a new `fix`, deriving from `FixMAD`, which already owns the
target, the gamma ramp, the loss, the diagnostics and the LAMMPS energy and
virial plumbing. You supply two methods.

```cpp
// fix_mad_myspectrum.h
#ifdef FIX_CLASS
FixStyle(mad/myspectrum,FixMADMySpectrum);
#else
class FixMADMySpectrum : public FixMAD {
 public:
  FixMADMySpectrum(class LAMMPS *, int, char **);
 protected:
  void compute_spectrum() override;             // fill y_pred
  void accumulate_bias(const double *dEdy) override;  // -dE/dr and the virial
};
#endif
```

In the constructor: read `arg[3]` as the property compute's ID, consume your
keywords (calling `parse_common()` first so the shared ones work), check
`prop->n_components()` is the width you need, build your abscissa and call
`finish_setup(grid)`.

`compute_spectrum()` must fill `y_pred` **identically on every rank** — the
residual drives the force, so a rank holding a partial spectrum builds a wrong
one. Call `prop->compute_property()` yourself and `MPI_Allreduce` your partial
sums.

`accumulate_bias(dEdy)` differentiates your spectrum back to one weight per
atom and calls `prop->adjoint(lambda, virial)`:

```cpp
// chain: dE/dy -> dE/du (the normalisation's adjoint) -> lambda_i
// then the backend turns lambda into forces and the stress
prop->adjoint(lambda, virial);
```

`fix_mad_xps.cpp` is the minimal example (a scalar, one Gaussian);
`fix_mad_nmr_aniso.cpp` is the same chain with one factor more (a tensor and
an orientation average). Register the style in `tools/register_madlib.py` and
document it in `README.md` and `doc/src/fix_mad.rst` in the same change.

---

## Five traps, each of which has produced a plausible wrong number here

**Ghost forces written in `post_force` are dropped.** `Verlet`
reverse-communicates *before* `post_force`, so a force written to a ghost there
is lost. `ComputeMADProperty` handles this for you: accumulate into `fadj`
(which covers `nlocal + nghost`) and call `scatter_adjoint()`. It matters in
serial too — periodic images are ghosts.

**The volume virial belongs to one rank.** Every rank adding it gives an
`nprocs`-fold stress.

**Monte Carlo re-prices inside one timestep.** Never cache a property or a
spectrum on the step number; a stale value accepts every trial move.

**A normalised spectrum's gradient is not the unnormalised one scaled.** For
`y = u/||u||` the adjoint carries a projection term,
`dE/du = (dE/dy - y (dE/dy . y)) / ||u||`, and dropping it leaves a gradient
that is wrong by a few percent and still points downhill.

**A finite quadrature is not invariant.** If your observable averages over
orientations (or anything else), an exact average cannot depend on how the
sample is oriented, so it exerts no torque — and what the finite grid leaves
behind is an antisymmetric part of `sum_i r_i (x) f_i`. It falls as 1/N and is
the sharpest available check that N is large enough.

## Testing it, cheapest first

A change is finished when a test that *can* go red says so.

```sh
LMP=<build>/lmp tests/run.sh <your leg>
```

1. **Against an independent computation of the same thing.** A numpy
   re-implementation of your spectrum from the dumped per-atom values, or the
   model framework's own pipeline. This is the only test that catches a wrong
   forward pass; a finite-difference gradient will happily confirm a
   self-consistently wrong one.
2. **Gradient by finite differences**, judged by the h-scan order (~2), not a
   tolerance: `tests/fd_gradient.py <data> <extra.lmp> --hs 1e-3,3e-4,1e-4`.
   A wrong constant factor gives order 0 and passes any single tolerance.
3. **Invariants**: `E = gamma D / 2`, `sum_j f_j = 0`.
4. **One rank against four**, energy, forces and virial.
5. **Several MD steps, not `run 0`.** `newton off` leaves ghost forces
   uncleared between steps, so `run 0` can be exact while step 1 is wrong by
   eV/Å.
6. **The negative control**: break the feature, rebuild, watch the test go red.
   A local-property gradient once passed here with its kernel deleted.

`tests/mad_tensor_toy.py` is a per-atom tensor with a closed form, for testing
a tensor-valued path without a model, and `tests/run.sh nmraniso` is a leg
doing the first five. The negative control is a hand operation: break it,
rebuild, run the leg, restore. Worth doing twice — once breaking the forward
pass and once breaking only the adjoint — because they fail different tests,
and a leg where both are checked by the same number is a leg with one test in
it.
