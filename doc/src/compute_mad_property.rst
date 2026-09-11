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
         *ncomp* value = 1 or 3 = components of the property
         *cutoff* value = neighbor list cutoff (distance units)

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

   mad_compute(positions, types, edge_i, edge_j, nlocal, ncomp) -> q
   mad_adjoint(positions, types, edge_i, edge_j, nlocal, ncomp, lam) -> f

with *q* of shape (nlocal,) or (nlocal, ncomp), and *f* of shape (nall, 3) equal
to the negative gradient of the weighted sum of *q* -- one backward pass for a
torch model.  The positions cover owned atoms and ghosts, and a ghost carries
the position of the periodic image it stands for, so the edge list and these
coordinates fix the geometry completely; no cell and no boundary conditions are
passed, and the model does not need to know about domain decomposition.  Arrays
cross as memoryviews over LAMMPS's own storage and nothing is copied.

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
