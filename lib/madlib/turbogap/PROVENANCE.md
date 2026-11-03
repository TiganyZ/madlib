# Where these files came from

Written by `tools/vendor_turbogap.sh`. Do not edit these sources by
hand -- an edited copy cannot be diffed against upstream, and that is
exactly how the previous set drifted without anyone noticing.

| | |
|---|---|
| vendored on | 2026-09-23T12:27:56Z |
| turbogap | `45b49a9d0074a28ca9c29cd8c7fe16cbd93a2cc3` |
| turbogap subject | removed extraneous gpu fixes |
| soap_turbo | `55ce9a95f3eaf5e386c2a74d977207ce3067f3a4` |
| soap_turbo subject | TURBOGAP_DUMP_RADIAL: do not index the derivative dummy |

## Files

| file | from | sha256 |
|---|---|---|
| `kinds.f90` | `src/` | `a92e10fe256904c9` |
| `elements.f90` | `src/` | `2e130fad2fb85ac0` |
| `topology.f90` | `src/` | `d01b718480cdfe51` |
| `splines.f90` | `src/` | `8b70bee192c7e592` |
| `types.f90` | `src/` | `6452dc3ab69cb34a` |
| `local_properties.f90` | `src/` | `a887581009a8e62a` |
| `gap.f90` | `src/` | `9fa54e003ae52062` |
| `soap_turbo_functions.f90` | `src/soap_turbo/src/` | `334e78ae7cd9638b` |
| `soap_turbo_radial.f90` | `src/soap_turbo/src/` | `5d90602ce7e5f8e1` |
| `soap_turbo_angular.f90` | `src/soap_turbo/src/` | `4eae57bd3df56662` |
| `soap_turbo_compress.f90` | `src/soap_turbo/src/` | `c1330805efc9aa79` |
| `soap_turbo.f90` | `src/soap_turbo/src/` | `c7e3279ef21b2453` |

Check they are still current with:

```sh
tools/vendor_turbogap.sh --check /path/to/turbogap
```
