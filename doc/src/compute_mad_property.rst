.. index:: compute mad/property/coord
.. index:: compute mad/property/python
.. index:: compute mad/property/dipole
.. index:: compute mad/property/mliap
.. index:: compute mad/property/table
.. index:: compute mad/property/turbogap

compute mad/property/coord command
==================================

compute mad/property/dipole command
===================================

compute mad/property/mliap command
==================================

compute mad/property/python command
===================================

compute mad/property/table command
==================================

compute mad/property/turbogap command
=====================================

Syntax
""""""

.. code-block:: LAMMPS

   compute ID group-ID style keyword values ...

* ID, group-ID are documented in :doc:`compute <compute>` command
* style = *mad/property/coord* or *mad/property/dipole* or *mad/property/mliap*
  or *mad/property/python* or *mad/property/table* or *mad/property/turbogap*

  .. parsed-literal::

       *mad/property/coord*:
         *cutoff* values = r1 r2 (distance units)

       *mad/property/dipole*:
         *charges* values = one charge per atom type (charge units)
         *charge* = take the charges from the atom style instead
         *neutral_tol* value = largest net charge tolerated

       *mad/property/mliap* keyword values
         *model* values = style file = *linear* or *quadratic* or *nn*, and its parameters
         *descriptor* values = style file = *sna* or *so3*, and its parameters

       *mad/property/python* args = module keyword values
         module = name of an importable Python module
         *ncomp* value = components of the property: 1 for a scalar, 3 for a
           dipole, 6 for a symmetric tensor in Voigt order
         *cutoff* value = neighbor list cutoff (distance units)
         *pbc* value = *yes* to pass one periodic cell instead of a ghost cloud
         *config* value = a string passed to the module's mad_bind()

       *mad/property/table*:
         *values* = one value per atom type
         *file* value = filename holding "atom-ID value" per line

       *mad/property/turbogap* args = file label keyword values
         file = a TurboGAP potential file
         label = name of the local property in it, e.g. core_electron_be
         *species* values = one element label per atom type

Examples
""""""""

.. code-block:: LAMMPS

   compute cebe all mad/property/turbogap CO.gap core_electron_be species C O
   compute mu   all mad/property/dipole charges 0.5 -0.5
   compute cn   all mad/property/coord cutoff 1.7 2.1
   compute q    all mad/property/table file cebe.dat
   compute qml  all mad/property/python my_mace_model ncomp 1 cutoff 5.0
   compute cebe all mad/property/python madlib_models ncomp 1 cutoff 12.0 &
                config "backend=mace model=cebe.model types=C,O"
   compute sig  all mad/property/python madlib_models ncomp 6 pbc yes &
                config "backend=nequip_nmr parameter=ms quantity=tensor6 model=MS/minimal types=Si,O stats=stats.json"
   compute qia  all mad/property/mliap model linear cebe.model descriptor sna cebe.descriptor

Description
"""""""""""

.. versionadded:: TBD

Calculate a per-atom quantity that an experimental observable depends on, and
its derivative with respect to the atom positions.  Observables such as an X-ray
photoelectron spectrum depend on the positions only through a per-atom scalar --
a core-electron binding energy -- and an infrared spectrum only through a
per-atom dipole.  These computes supply that quantity to
:doc:`fix mad/xps <fix_mad>` and :doc:`fix mad/ir <fix_mad_ir>`.

What such a fix needs back is not the full Jacobian
:math:`\partial q_i / \partial \mathbf{r}_j` but its contraction with a weight,

.. math::

   \mathbf{f}_j \mathrel{-}= \sum_i \lambda_i
       \frac{\partial q_i}{\partial \mathbf{r}_j}

which is the force any model already computes from a per-atom energy
:math:`\sum_i \lambda_i q_i` with the weights held fixed.  A new backend
therefore needs a forward pass and that one contraction, which is why these
styles can wrap models of very different kinds.

*mad/property/turbogap* evaluates a Gaussian approximation potential fitted with
TurboGAP,

.. math::

   q_i = V_0 + \delta^2 \sum_s \alpha_s\,
         (\mathbf{q}_i \cdot \mathbf{Q}_s)^\zeta

reading an existing potential file unchanged.  A model is normally one block per
central species, since different elements have different reference energies.
The *species* keyword gives the element label of each atom type.

*mad/property/dipole* gives :math:`\boldsymbol{\mu}_i = q_i \mathbf{r}_i` from
per-type charges or from the charges carried by the atom style, so any model
that writes charges is a dipole model here.  **The sum of** :math:`q_i
\mathbf{r}_i` **is not defined in a periodic box**: it depends on the origin and
on which periodic images the atoms happen to be wrapped into.  A periodic run
therefore requires an atom style with molecule IDs, and the dipole is then the
sum of molecular dipoles built from intramolecular vectors, which does not move
with the origin.  A periodic box without molecule IDs is an error.

.. note::

   This is not the dipole TurboGAP predicts.  TurboGAP uses no charges: its
   dipole comes from a fitted model, a ``soap_turbo`` block flagged
   ``dipole_model = .true.`` whose local energy is a fictitious scalar fitted so
   that its gradient with respect to the central atom's own position is the
   local dipole,

   .. math::

      \boldsymbol{\mu}_i = \frac{\partial E_i}{\partial \mathbf{r}_i}
        = \zeta\delta^2 \sum_s \alpha_s\,
          (\mathbf{q}_i \cdot \mathbf{Q}_s)^{\zeta-1}\,
          \frac{\partial \mathbf{q}_i}{\partial \mathbf{r}_i}

   This is the energy kernel's own contraction taken at the *self* pair of each
   site rather than scattered over its neighbours, and with :math:`+\zeta\delta^2`
   rather than :math:`-\zeta\delta^2`, a dipole being a gradient where a force
   is its negative.  It is fitted to reference dipoles and is a different
   quantity from a charge model's, not a coarser version of one.  No style here
   evaluates it yet: *mad/property/turbogap* reads local-property blocks, and
   given a dipole model it stops and says so.  A potential file flagged
   ``dipole_model`` is likewise refused by ``pair_style turbogap``, since the
   energy in it is a fitting artefact that would otherwise be reported as a
   perfectly ordinary number.

*mad/property/mliap* evaluates an :doc:`ML-IAP <pair_mliap>` model.  Such a
model maps descriptors to a per-atom scalar; that scalar is normally an energy,
but nothing in the machinery requires it to be, and the same linear, quadratic
or neural-network model fitted against core-electron binding energies is a
binding-energy model.  Every descriptor ML-IAP supports comes with it.

The adjoint costs no extra descriptor pass: ML-IAP already forms
:math:`\beta_{ik} = \partial q_i/\partial B_{ik}` and contracts it against
the descriptor derivatives to make forces, so scaling the betas by
:math:`\lambda_i` beforehand turns that same contraction into what the bias
needs.

*mad/property/python* calls a model written in Python, so any framework
reachable from there is a backend: MACE, GRACE, a torch module, a plain numpy
expression.  The module must provide two functions,

.. code-block:: python

   mad_compute(positions, types, edge_i, edge_j, nlocal, ncomp, cell) -> q
   mad_adjoint(positions, types, edge_i, edge_j, nlocal, ncomp, lam, cell) -> f

with *q* of shape (nlocal,) or (nlocal, ncomp), and *f* the negative gradient of
the weighted sum of *q* -- one backward pass for a torch model.  Arrays cross as
memoryviews over LAMMPS's own storage and nothing is copied.

A module configured only by its import can serve one model and one quantity per
run, which is a poor fit for a deck that wants a model's binding energies *and*
its dipoles.  So a module may instead provide

.. code-block:: python

   mad_bind(config) -> (compute_fn, adjoint_fn)

where *config* is the string given to the *config* keyword and the two returned
callables have the signatures above.  Each compute binds separately and gets its
own closed-over state.  A *config* string handed to a module without
``mad_bind`` is an error, not an ignored argument: a model left on its
import-time default still returns numbers.

The MADLIB repository ships such a module, ``madlib_models`` in ``python/``,
which is the general bridge from a machine-learned model to an observable: it
loads the model, runs it on the geometry LAMMPS hands over, reduces its output
to the quantity the observable wants, and differentiates.  Core-electron
binding energies from a MACE model into :doc:`fix mad/xps <fix_mad>`, a dipole
into :doc:`fix mad/ir <fix_mad_ir>`, a shielding tensor into
:doc:`fix mad/nmr/aniso <fix_mad>`, all without writing a line of Python:

.. code-block:: LAMMPS

   compute cebe all mad/property/python madlib_models ncomp 1 cutoff 12.0 &
               config "backend=mace model=cebe.model types=C,O"

Its keys are *backend* (*mace*, *nequip_nmr* or *callable*), *model*, *types*
-- the element of each LAMMPS atom type -- *output*, which of the model's
outputs to read, *quantity*, how to reduce it (*scalar*, *vector*, *tensor6*,
*iso*, *span*, *vzz*, *etaq*, *component:zz*), and *scale*/*shift* for a unit
or sign convention.  Models are cached by what identifies them, so two
computes over one model -- an isotropic shielding and the span of the same
tensor -- cost one forward pass per step and two backward passes.  The package
docstring lists the rest.

There are two ways the module can be told where the atoms are, and *cell* says
which it has been given.  By default *cell* is nine zeros, the positions cover
owned atoms **and their ghosts** -- each ghost carrying the position of the
periodic image it stands for -- and *f* is (nall, 3).  The edge list and those
coordinates fix the geometry completely: no cell and no boundary conditions are
needed, the model knows nothing about domain decomposition, and forces returned
for ghosts are communicated back to their owners.  This is the general case and
the only one that runs under MPI.  Its *cutoff* must reach the model's whole
interaction range, which for a message-passing model is the neighbor-list cutoff
times the number of layers, and the communication cutoff must cover that.

.. warning::

   Even then it is exact only for a model that passes **one** message.  LAMMPS
   builds neighbor lists for owned atoms, so every edge handed over ends on an
   owned atom and a ghost receives no messages of its own.  A second layer
   therefore reads an incomplete feature for every neighbor that is a ghost.
   Measured on a 91-atom cell with a two-layer MACE at *r_max* 2 Angstrom and a
   4 Angstrom shell, the error is 6e-4 eV per atom; the same model with one
   layer agrees with MACE's own pipeline to 4e-16.  *pbc yes* has no such
   error, because there the model builds its own graph.

With *pbc yes*, *cell* holds the three cell vectors, the positions are the owned
atoms only, and a model that handles periodicity itself is left to.  This is for
a long-ranged model on a small cell, where the ghost route is not merely slower:
ShiftML passes two messages over a 5 Angstrom neighbor list, so a 28-atom
molecular crystal needs a 10 Angstrom ghost shell, which is 3304 atoms -- 118
copies of the cell to ask about 28 of them, and twenty times the wall time of
passing the cell.  The two agree to :math:`10^{-4}` ppm, the model's own float32
noise.  It runs on one rank, in a periodic box, over the *all* group, and says
so rather than quietly meaning something else.

In that mode *f* is (nlocal + 3, 3): the last three rows are
:math:`-\partial E/\partial \mathbf{h}`, one per cell vector.  They are needed
because the periodic images that carry the rest of :math:`\sum \mathbf{r}
\otimes \mathbf{f}` are inside the model rather than in the atom arrays, so
without them the pressure would be wrong rather than absent.  For a torch model
they are the second output of the same backward pass.

*mad/property/coord* gives a smooth coordination number, one below *r1*, a
raised cosine to *r2*, and zero beyond.  It has an analytic gradient and needs no
model files, so it is useful for checking a bias end to end before a fitted model
is involved.

*mad/property/table* gives values read from a file or set per type.  Its gradient
is zero by construction, so it can predict a spectrum but cannot drive a bias;
asking it for forces is an error rather than a silently absent force.

Output info
"""""""""""

These computes calculate a per-atom vector, or a per-atom array with three
columns for *mad/property/dipole*.  The values are "intensive", in whatever units
the quantity has: energy units for a binding energy, charge-distance for a
dipole, and unitless for a coordination number.

Restrictions
""""""""""""

These computes are part of the MADLIB package.  They are only enabled if LAMMPS
was built with that package.  See the :doc:`Build package <Build_package>` doc
page for more info.

*mad/property/turbogap* additionally requires the optional Fortran backend in
``lib/madlib``, selected with ``-D MADLIB_ENABLE_TURBOGAP=on``, and
*mad/property/python* the embedded interpreter, selected with
``-D MADLIB_ENABLE_PYTHON=on``.  *mad/property/mliap* requires the ML-IAP
package and the package supplying its descriptor.  Without them the styles still exist and report
what to build.

That Fortran backend has two implementations, and a build gets one of them:
adding ``-D MADLIB_ENABLE_GPU=on`` computes the SOAP descriptor and the
sparse-GP prediction on the device instead of the host, from an external
TurboGAP checkout.  The style itself is unchanged, and so is the answer: the
device predictor applies the same *zero_trunc* clamp (core electron binding
energies and Hirshfeld volumes both set it), flooring the same sites and
dropping their gradients.  That clamp is recent in TurboGAP, and configuring
against a checkout without it is refused rather than silently answered
differently.  See the MADLIB repository's ``docs/TURBOGAP-KK.md``.

For *mad/property/coord*, *mad/property/python* and *mad/property/turbogap*,
the cutoff plus the
neighbor skin must be within the communication cutoff; when it is not, the
compute stops with a message naming the :doc:`comm_modify <comm_modify>`
*cutoff* option.

Related commands
""""""""""""""""

:doc:`fix mad/xps <fix_mad>`, :doc:`fix mad/ir <fix_mad_ir>`,
:doc:`compute coord/atom <compute_coord_atom>`,
:doc:`compute dipole <compute_dipole>`

Default
"""""""

For *mad/property/dipole*, *neutral_tol* = 1.0e-6.  The other styles have no
optional settings with defaults.
