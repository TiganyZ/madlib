.. index:: compute mad/pdf

compute mad/pdf command
=======================

Syntax
""""""

.. code-block:: LAMMPS

   compute ID group-ID mad/pdf r rmin rmax nbins keyword values ...

* ID, group-ID are documented in :doc:`compute <compute>` command
* mad/pdf = style name of this compute command
* r values = rmin rmax nbins = range and number of bins (distance units)
* zero or more keyword/value pairs may be appended

  .. parsed-literal::

       *kde* value = width of the Gaussian each pair distance is smeared with
                     (distance units)
       *rcut* value = neighbor list cutoff (distance units)
       *trunc* value = drop Gaussian tails past this many widths
       *compat* value = *turbogap* = reproduce TurboGAP's shell volume

Examples
""""""""

.. code-block:: LAMMPS

   compute gr all mad/pdf r 0.1 6.0 201 kde 0.1
   fix     out all ave/time 1 1 100 c_gr[*] mode vector file gr.dat

Description
"""""""""""

.. versionadded:: TBD

Calculate the partial pair distribution functions :math:`g_{ab}(r)`, in a form
that can be differentiated with respect to the atom positions:

.. math::

   g_{ab}(r_l) = \frac{V}{N_a N_b f_{ab}}
                 \frac{1}{\sqrt{2\pi}\,\sigma\,4\pi r_l^2}
                 \sum_{i \in a, j \in b} e^{-(r_l - r_{ij})^2 / 2\sigma^2}

with :math:`f_{ab}` equal to one for a like pair and two otherwise.  This is
the same quantity :doc:`compute rdf <compute_rdf>` reports, resolved into
species pairs and smeared with a Gaussian rather than binned.

The Gaussian is what makes it differentiable.  A histogram has zero gradient
almost everywhere, so a bias built on one would apply no force; a zero *kde*
width is therefore an error rather than a silently inert setting.

By default the neighbor list cutoff sits *trunc* widths beyond *rmax*, so that a
pair crossing it changes the pattern by less than round-off.  Setting *rcut*
smaller brings the cut inside the range the Gaussians reach, which steps the
pattern as pairs cross it.  That is invisible in :math:`g(r)` itself but makes
the virial of any fix consuming this compute wrong, so it warns.

This compute is the pair-distribution input of :doc:`fix mad/pdf, fix mad/sq and
fix mad/xrd <fix_mad>`, and several of those fixes can share one instance.  It
is equally usable on its own, to report :math:`g(r)`.

Output info
"""""""""""

This compute calculates a global array with *nbins* rows and one column more
than the number of species pairs.  The first column is the bin centre; the
remaining columns are the partials, ordered over the upper triangle of species
pairs, row by row: (1,1), (1,2), ... (1,n), (2,2), and so on.  The values are
"intensive".

Restrictions
""""""""""""

This compute is part of the MADLIB package.  It is only enabled if LAMMPS was
built with that package.  See the :doc:`Build package <Build_package>` doc page
for more info.

The cutoff plus the neighbor skin must be within the communication cutoff; when
it is not, the compute stops with a message naming the
:doc:`comm_modify <comm_modify>` *cutoff* option.  A cutoff larger than the
force cutoff requires :doc:`neighbor <neighbor>` style *bin* or *nsq*.

Related commands
""""""""""""""""

:doc:`compute rdf <compute_rdf>`, :doc:`fix mad/pdf <fix_mad>`

Default
"""""""

*rcut* = rmax + *trunc* :math:`\times` *kde*, *trunc* = 8.0, *compat* = none.
The *kde* width has no default and must be given.
