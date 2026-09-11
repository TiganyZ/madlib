.. index:: fix mad/pdf
.. index:: fix mad/sq
.. index:: fix mad/xrd
.. index:: fix mad/xps

fix mad/pdf command
===================

fix mad/sq command
==================

fix mad/xrd command
===================

fix mad/xps command
===================

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID style compute-ID keyword values ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* style = *mad/pdf* or *mad/sq* or *mad/xrd* or *mad/xps*
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

       *mad/xps*:
         *energy* values = emin emax nsamples
         *sigma* value = broadening width of each per-atom value (energy units)
         *norm* value = *l2* or *area* or *none*
         *trunc* value = drop Gaussian tails past this many sigma

Examples
""""""""

.. code-block:: LAMMPS

   compute gr all mad/pdf r 0.1 6.0 201 kde 0.1
   fix xrd all mad/xrd c_gr target measured.fq q 1.05 9.97 201 &
                       output q*F(q) species C O gamma 0.0 100.0
   fix_modify xrd energy yes virial yes

   compute cebe all mad/property/turbogap CO.gap core_electron_be species C O
   fix xps all mad/xps c_cebe target xps.dat energy 280 300 201 sigma 0.4 gamma 100.0

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
*trunc* = 8.0.
