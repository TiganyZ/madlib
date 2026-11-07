.. index:: fix mad/pdf
.. index:: fix mad/sq
.. index:: fix mad/xrd
.. index:: fix mad/xps
.. index:: fix mad/nmr
.. index:: fix mad/nmr/aniso

fix mad/pdf command
===================

fix mad/sq command
==================

fix mad/xrd command
===================

fix mad/xps command
===================

fix mad/nmr command
===================

fix mad/nmr/aniso command
=========================

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID style compute-ID keyword values ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* style = *mad/pdf* or *mad/sq* or *mad/xrd* or *mad/xps* or *mad/nmr* or
  *mad/nmr/aniso*
* compute-ID = ID of the :doc:`compute mad/pdf <compute_mad_pdf>` or
  :doc:`compute mad/property <compute_mad_property>` command that supplies the
  prediction
* zero or more keyword/value pairs may be appended

  .. parsed-literal::

       shared by every *mad* style:
         *target* value = filename
           filename = two columns, abscissa and measured intensity, no header
         *gamma* values = k or k_begin k_end
           k = weight of the mismatch in the energy (energy units)
           k_begin k_end = ramp the weight linearly over the run
         *every* value = N = apply the bias every N steps
         *fit* values = scale offset
           scale, offset = *yes* or *no*, fit them by weighted least squares
         *file* value = filename for the predicted spectrum
         *write_every* value = N = write that file every N steps
         *compat* value = *turbogap* = reproduce TurboGAP's conventions

       *mad/pdf*:
         *output* value = *pdf* or *D(r)*

       *mad/sq* and *mad/xrd*:
         *q* values = qmin qmax nq
         *units* value = *q* (inverse length) or *twotheta* (degrees)
         *weighting* value = *sq* or *xray* or *neutron*
         *species* values = one element label per atom type
         *output* value = *raw* or *F(q)* or *q\*F(q)*
         *wavelength* value = incident wavelength (distance units)
         *window* value = *lorch* or *none*
         *window_cut* value = cutoff of that window (distance units)
         *lp* value = *yes* or *no* = powder Lorentz-polarization factor
         *polarization* value = degree of polarization P of the incident beam

       *mad/xps*, *mad/nmr* and *mad/nmr/aniso*:
         *energy* (or *range*) values = min max nsamples
         *sigma* value = broadening width of each per-atom value
         *norm* value = *l2* or *area* or *none*
         *trunc* value = drop Gaussian tails past this many sigma

       *mad/nmr* and *mad/nmr/aniso*, and one of the two is required:
         *reference* value = a = convert a shielding to a shift, delta = a - sigma
         *slope* value = b = use delta = a - b sigma instead
         *convention* value = *shift* = the property is already on the target's axis

       *mad/nmr/aniso* only:
         *orientations* value = N = crystallite orientations in the powder average

Examples
""""""""

.. code-block:: LAMMPS

   compute gr all mad/pdf r 0.1 6.0 201 kde 0.1
   fix xrd all mad/xrd c_gr target measured.fq q 1.05 9.97 201 &
                       output q*F(q) species C O gamma 0.0 100.0
   fix_modify xrd energy yes virial yes

   compute cebe all mad/property/turbogap CO.gap core_electron_be species C O
   fix xps all mad/xps c_cebe target xps.dat energy 280 300 201 sigma 0.4 gamma 100.0

   group   protons type 1
   compute cs all mad/property/python mad_shiftml ncomp 1 pbc yes
   fix h1 protons all mad/nmr c_cs target h1.dat range 0 20 201 sigma 0.5 &
                              reference 30.96 gamma 50.0

   group   si all type 1
   compute sig all mad/property/python madlib_models ncomp 6 pbc yes &
               config "backend=nequip_nmr parameter=ms quantity=tensor6 &
                       model=models/MS/final/isd/minimal types=Si,O &
                       stats=data_mean_std.json"
   fix si29 si all mad/nmr/aniso c_sig target si29.dat range -140 -80 201 &
                   sigma 2.0 orientations 1024 reference 328.4 gamma 20.0

Description
"""""""""""

.. versionadded:: TBD

Bias a simulation towards experimental data.  Each of these fixes predicts an
observable from the current configuration, compares it with a measured curve,
and adds a penalty for the difference to the potential energy:

.. math::

   V_\mathrm{tot} = V_\mathrm{pot} + \frac{k}{2} \sum_i w_i
                    \left( s\,y_i(\{\mathbf{r}\}) + b - y^\mathrm{exp}_i \right)^2

The force is the gradient of that penalty, so the simulation is pulled towards
configurations whose predicted observable matches the measurement.  This is
sometimes called molecular augmented dynamics, and the Monte Carlo form of it
reverse Monte Carlo.  The weights :math:`w_i` are trapezoidal in the abscissa
and normalised to mean one, so *gamma* means the same thing whatever grid the
measurement was sampled on.

*mad/pdf* fits a pair distribution function, *mad/sq* a structure factor, and
*mad/xrd* an X-ray or neutron diffraction pattern; all three read the partial
pair distributions from a :doc:`compute mad/pdf <compute_mad_pdf>`.  *mad/xps*
fits a spectrum built by broadening a per-atom quantity supplied by a
:doc:`compute mad/property <compute_mad_property>` command, such as a
core-electron binding energy; it is also registered as *mad/spectrum*, since
nothing in it is specific to photoemission.

The *gamma* keyword sets how hard the data pulls against the potential.  Given
two values it ramps linearly over the run, which is how a restraint is switched
on without disturbing the structure at the first step.  **Size it from the
unbiased mismatch**: run once with *gamma* = 0.0, read the dissimilarity from
the third element of the fix's global vector, and choose *gamma* so that the
bias energy is a small fraction of the cohesive energy.  A weight too large by
orders of magnitude will pull the structure apart while the reported mismatch
falls, which does not look like a failure.

Predicted intensities are often in arbitrary units.  The *fit* keyword fits an
overall scale, and optionally an additive baseline, by weighted least squares
before comparing.  Because both sit at their own optimum, the gradient of the
penalty picks up no extra term from them.  A fitted scale that comes out
negative would reverse the bias, so it is rejected and the scale-only fit used
instead.

For *mad/sq* and *mad/xrd*, *species* gives the element label of each atom type,
which selects the X-ray form factors or neutron scattering lengths.  The *window*
keyword applies a Lorch window that damps the ringing caused by truncating the
transform; its cutoff is the extent of the *r* grid, which is not the same
number as the neighbor list cutoff of the compute.

*mad/nmr* broadens chemical shifts exactly as *mad/xps* broadens binding
energies, and differs in two ways that are both about NMR rather than about the
arithmetic.

**Which nucleus is the fix's group.** A :math:`^{1}\mathrm{H}` and a
:math:`^{13}\mathrm{C}` spectrum are two fixes over two groups reading one
compute; the compute still covers every atom, because the model needs the whole
structure to predict any part of it.

**Shielding is not shift, and the difference is a reflection.**  A model
predicts a shielding :math:`\sigma`; an experiment reports a shift
:math:`\delta = a - b\sigma`, with *b* near one.  Broadening shieldings
against a shift target therefore compares a spectrum with its own mirror image,
and the *fit* keyword will obligingly rescale that into something that looks
like a fit.  So *mad/nmr* will not start until told which it has been handed:
*reference a* (with *slope b*, default one) converts, and *convention shift*
says the property is already a shift.  There is no default, deliberately.

Referencing constants are specific to the model and the level of theory it was
fitted at, and taking one from a different source is a common way to get a
spectrum that is uniformly wrong.  When the "measured" data is itself a
prediction from the same model -- as in a test, or a structure search against a
reference structure's computed spectrum -- both sides are shieldings and
*convention shift* is the honest setting.

*mad/nmr/aniso* fits a **static powder lineshape**, which is what a shielding
tensor buys over its trace.  *mad/nmr* broadens one number per nucleus, so
every site enters the spectrum as a symmetric peak and the anisotropy of its
environment is discarded before the comparison.  A static powder spectrum is
that anisotropy: a crystallite whose field direction is :math:`\mathbf{n}`
resonates at

.. math::

   \omega_i(\mathbf{n}) = \mathbf{n} \cdot \mathbf{T}_i \cdot \mathbf{n}

and the spectrum is that over all orientations, which is a skewed lineshape
with edges at the principal values rather than a peak at their mean.  Two
sites with the same isotropic shielding and different tensors are then two
different patterns.

The property it reads is a **symmetric tensor in Voigt order**, so the compute
must have *ncomp* 6 and its six components must be

.. parsed-literal::

   T_xx  T_yy  T_zz  T_xy  T_xz  T_yz

with no factor of two folded into the off-diagonal ones.  An NMR experiment
cannot see the antisymmetric part of the tensor to first order, so a model
predicting a general rank-2 tensor should symmetrise it; the *quantity*
*tensor6* of the :doc:`madlib_models <compute_mad_property>` bridge does
exactly that.  Nothing in the fix diagonalises anything -- the orientation
average needs no principal values, no ordering convention and no eigenvalue
derivatives, so there is no branch for a degenerate site to cross.

**Orientations are a convergence parameter.**  A powder pattern has edges at
the principal values, and too coarse a quadrature resolves them into a comb of
Gaussians which is smooth enough to look like a spectrum.  The default grid is
a golden spiral over the hemisphere -- a hemisphere because
:math:`\omega(\mathbf{n}) = \omega(-\mathbf{n})` -- and the check is to raise
*orientations* until the lineshape stops moving at the *sigma* in use.

There is a second, sharper way to see the same thing.  An exact powder average
cannot depend on how the sample is oriented, so it exerts no torque; a finite
grid does, and the leftover appears as an antisymmetric part of
:math:`\sum_i \mathbf{r}_i \otimes \mathbf{f}_i`.  The reported stress is
its symmetric half, and the asymmetry falls as :math:`1/N` -- 2 percent of the
shear at 48 orientations, 0.1 percent at 3072 in the package's own test.  If a
run's angular momentum drifts, that number is why.

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about these fixes is written to :doc:`binary restart files
<restart>`.

The :doc:`fix_modify <fix_modify>` *energy* and *virial* options are supported,
and are enabled by default, so the bias contributes to the total potential
energy and to the pressure.  Because the energy is included in
:doc:`compute pe <compute_pe>`, the bias is seen by the Metropolis test of
:doc:`fix gcmc <fix_gcmc>`, :doc:`fix atom/swap <fix_atom_swap>` and
:doc:`fix widom <fix_widom>` without further settings, which is what makes an
experimentally biased Monte Carlo run possible.

These fixes compute a global scalar, the bias energy, and a global vector of
length 6: the bias energy, the relative error, the dissimilarity, the current
weight, the fitted scale and the fitted offset.  The relative error is
dimensionless and is the number to watch during a run.  The scalar and vector
values are "intensive".

These fixes are invoked during :doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

These fixes are part of the MADLIB package.  They are only enabled if LAMMPS was
built with that package.  See the :doc:`Build package <Build_package>` doc page
for more info.

They require :doc:`units <units>` *metal* or *real*.

*mad/xps* with a per-atom property from a model requires that model's backend to
be available; see :doc:`compute mad/property <compute_mad_property>`.

Related commands
""""""""""""""""

:doc:`compute mad/pdf <compute_mad_pdf>`,
:doc:`compute mad/property <compute_mad_property>`,
:doc:`fix mad/ir <fix_mad_ir>`,
:doc:`compute rdf <compute_rdf>`,
:doc:`fix gcmc <fix_gcmc>`,
:doc:`fix atom/swap <fix_atom_swap>`

Default
"""""""

*gamma* = 0.0, *every* = 1, *fit* = no no, *compat* = none.
For *mad/pdf*, *output* = pdf.  For *mad/sq*, *weighting* = sq and *output* =
raw; for *mad/xrd*, *weighting* = xray and *output* = q\*F(q); for both,
*units* = q, *window* = lorch, *lp* = no, *polarization* = 1.0 and
*wavelength* = 1.5405981.  For *mad/xps*, *sigma* = 0.4, *norm* = l2 and
*trunc* = 8.0.  For *mad/nmr/aniso*, *sigma* = 1.0, *orientations* = 512,
*norm* = l2 and *trunc* = 8.0.
