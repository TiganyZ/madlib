# Where these files came from

Written by `tools/vendor_turbogap.sh`. Do not edit these sources by
hand -- an edited copy cannot be diffed against upstream, and that is
exactly how the previous set drifted without anyone noticing.

| | |
|---|---|
| vendored on | 2026-09-12T08:27:43Z |
| turbogap | `84d503b9184d62da02575d2f28e0b8f064fbf4a8` |
| turbogap subject | Merge feature/kokkos-backend: a second device backend, behind KOKKOS=1 |
| soap_turbo | `55ce9a95f3eaf5e386c2a74d977207ce3067f3a4` |
| soap_turbo subject | TURBOGAP_DUMP_RADIAL: do not index the derivative dummy |

## Files

| file | from | sha256 |
|---|---|---|
| `kinds.f90` | `src/` | `a92e10fe256904c9` |
| `local_properties.f90` | `src/` | `8b13cceb9ec5cc6a` |
| `soap_turbo_functions.f90` | `src/soap_turbo/src/` | `334e78ae7cd9638b` |
| `soap_turbo_radial.f90` | `src/soap_turbo/src/` | `5d90602ce7e5f8e1` |
| `soap_turbo_angular.f90` | `src/soap_turbo/src/` | `4eae57bd3df56662` |
| `soap_turbo_compress.f90` | `src/soap_turbo/src/` | `c1330805efc9aa79` |
| `soap_turbo.f90` | `src/soap_turbo/src/` | `c7e3279ef21b2453` |

Check they are still current with:

```sh
tools/vendor_turbogap.sh --check /path/to/turbogap
```
