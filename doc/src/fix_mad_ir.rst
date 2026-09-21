.. index:: fix mad/ir

fix mad/ir command
==================

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID mad/ir N compute-ID keyword values ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* mad/ir = style name of this fix command
* N = store the total dipole every N steps
* compute-ID = ID of a :doc:`compute mad/property <compute_mad_property>` command
  with three components, giving a per-atom dipole
* zero or more keyword/value pairs may be appended

  .. parsed-literal::

       *nu_max* value = top of the returned range (inverse cm)
       *acf_ratio* value = fraction of the stored frames kept as lags
       *window* value = *hann* or *bartlett* or *welch* or *lorch* or *blackman* or *none*
       *qc* value = *harmonic* or *classical* or *linear* or *none*
       *temp* value = temperature for the *harmonic* correction (temperature units)
       *smooth* values = k kind
         k = smoothing width in bins
         kind = *gaussian* or *box*
       *subtract_mean* value = *yes* or *no*
       *normalise* value = *yes* or *no*
       *dc_cutoff* value = exclude bins below this from the power normaliser (inverse cm)
       *frame_dt* value = interval between stored frames (fs), for a :doc:`rerun <rerun>`
       *file* value = filename for the spectrum
       *write_every* value = N = write that file every N steps

Examples
""""""""

.. code-block:: LAMMPS

   compute mu all mad/property/dipole charges 0.5 -0.5
   fix     ir all mad/ir 2 c_mu nu_max 4000 acf_ratio 0.25 qc classical &
                         smooth 5 gaussian file ir_spectrum.dat

Description
"""""""""""

.. versionadded:: TBD

Predict an infrared spectrum from the total dipole of the trajectory.  The
dipole is stored every *N* steps, and its autocorrelation is transformed:

.. math::

   C(\tau) = \langle \delta\boldsymbol{\mu}(t) \cdot
                     \delta\boldsymbol{\mu}(t+\tau) \rangle, \qquad
   I(\nu) = P(\nu) \sum_\tau a(\tau) \cos(2\pi c\,\nu\,\tau)

The correlation is computed by the Wiener-Khinchin theorem over the whole
stored buffer.  This fix predicts a spectrum; it applies no force.

**The dipole is whatever the named compute produces**, and the only style that
produces one today is :doc:`compute mad/property/dipole
<compute_mad_property>`, which builds it from partial charges.  TurboGAP
instead drives its IR from a fitted dipole model, a different quantity; see the
note on that doc page.  The estimator here reproduces TurboGAP's ``ir_fft`` to
machine precision, but that comparison gives both codes the same stored
dipoles, so it verifies the spectrum and not the dipole feeding it.

Two of its settings surprise people often enough that both are written into the
header of the output file.

**The resolution is set by the longest lag, not by the run length.**
*acf_ratio* is the fraction of the stored frames kept as lags, and the bin
spacing follows from it.  A long run at a small *acf_ratio* gives a coarse
spectrum, and sampling for longer does not change that.

**The weighting is a physical choice.**  *harmonic* applies
:math:`\nu(1-e^{-hc\nu/kT})`, *classical* applies :math:`\nu^2`.  These agree
only when :math:`hc\nu \ll kT`, which at 300 K means below about 200
inverse cm; higher up they differ by more than an order of magnitude.  The
difference is a tilt across the range, so no overall scale absorbs it.  When two
spectra of the same trajectory disagree about band heights, check this first.

The *frame_dt* keyword exists for a :doc:`rerun <rerun>`, where the timestep is
not the interval between the frames being read.

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about this fix is written to :doc:`binary restart files
<restart>`, so a continued run starts a fresh buffer.

This fix computes a global scalar, the number of stored frames; a global vector
of length 5 holding the frames, the lags, the bin spacing, the Nyquist limit and
the peak intensity; and a global array with one row per frequency and three
columns, the wavenumber, the intensity and the power.  All are "intensive".

Restrictions
""""""""""""

This fix is part of the MADLIB package.  It is only enabled if LAMMPS was built
with that package.  See the :doc:`Build package <Build_package>` doc page for
more info.

It requires :doc:`units <units>` *metal* or *real*, since a wavenumber axis has
no meaning without a known time unit.  Requesting a *nu_max* above the Nyquist
limit of the sampling interval is an error.

Related commands
""""""""""""""""

:doc:`compute mad/property <compute_mad_property>`,
:doc:`fix mad/pdf <fix_mad>`,
:doc:`fix ave/correlate <fix_ave_correlate>`

Default
"""""""

*nu_max* = 4000.0, *acf_ratio* = 0.1, *window* = hann, *qc* = harmonic,
*temp* = 300.0, *smooth* = 10 gaussian, *subtract_mean* = yes,
*normalise* = yes, *dc_cutoff* = 100.0.
