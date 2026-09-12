#!/usr/bin/env bash
#
# Refresh the vendored TurboGAP sources under lib/madlib/turbogap.
#
# Why these files are vendored at all: MADLIB needs soap_turbo and TurboGAP's
# local-property GPR, and nothing else from TurboGAP. Adding the whole code as a
# submodule would drag in its driver, its MPI layer and two more submodules of
# its own, to reach seven files. So they are copied.
#
# The cost of copying is drift, and drift that nobody can see is the worst kind:
# before this script existed there was no record of WHICH TurboGAP the copies
# came from, so "is this current?" could only be answered by diffing seven files
# by hand against a checkout you had to find first. PROVENANCE.md now answers it
# in one line, and this script is the only thing that should ever write these
# files.
#
# The copies are VERBATIM. local_properties.f90 carries `#ifdef _GPU` blocks
# that need TurboGAP's device layer; they compile out because the build passes
# -cpp and never defines _GPU. Do not hand-strip them -- an edited copy cannot
# be diffed against upstream, which is how the last one drifted.
#
# Usage:
#     tools/vendor_turbogap.sh /path/to/turbogap
#     tools/vendor_turbogap.sh --check /path/to/turbogap    # report, change nothing
#
# The path is a TurboGAP checkout with its submodules initialised.

set -eu

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/.." && pwd)

# ---------------------------------------------------------------- parameters

DEST=$repo/lib/madlib/turbogap
PROVENANCE=$DEST/PROVENANCE.md

# Seven files, and where each lives in a TurboGAP checkout. soap_turbo is a
# submodule of it, so two different source directories are involved.
FILES_SRC="kinds.f90 local_properties.f90"
FILES_SOAP="soap_turbo_functions.f90 soap_turbo_radial.f90 soap_turbo_angular.f90 soap_turbo_compress.f90 soap_turbo.f90"

check_only=0
if [ "${1:-}" = "--check" ]; then
    check_only=1
    shift
fi

# ----------------------------------------------------------------- functions

say() { printf '%s\n' "$*"; }

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

check_source_tree() {
    local tg=$1
    [ -d "$tg" ] || die "no such directory: $tg"
    [ -f "$tg/src/kinds.f90" ] || die "$tg does not look like a TurboGAP checkout (no src/kinds.f90)"
    [ -f "$tg/src/soap_turbo/src/soap_turbo.f90" ] ||
        die "$tg/src/soap_turbo is empty -- run: git submodule update --init --recursive"
    return 0
}

# Read a commit, or say so plainly when the directory is not a git checkout,
# rather than writing "unknown" into the record as though it were a version.
#
# TURBOGAP_COMMIT and SOAP_TURBO_COMMIT override, for the case this was written
# for: the authoritative checkout lives on a cluster and what you have locally
# is a copy of its src/ with no .git beside it. Recording the real commit then
# needs it passed in, and a recorded commit that is a guess is worse than none.
commit_of() {
    local dir=$1
    case "$dir" in
        */soap_turbo) [ -n "${SOAP_TURBO_COMMIT:-}" ] && { echo "$SOAP_TURBO_COMMIT"; return 0; } ;;
        *) [ -n "${TURBOGAP_COMMIT:-}" ] && { echo "$TURBOGAP_COMMIT"; return 0; } ;;
    esac
    if git -C "$dir" rev-parse HEAD >/dev/null 2>&1; then
        git -C "$dir" rev-parse HEAD
    else
        echo "NOT-A-GIT-CHECKOUT"
    fi
}

subject_of() {
    local dir=$1
    case "$dir" in
        */soap_turbo) [ -n "${SOAP_TURBO_SUBJECT:-}" ] && { echo "$SOAP_TURBO_SUBJECT"; return 0; } ;;
        *) [ -n "${TURBOGAP_SUBJECT:-}" ] && { echo "$TURBOGAP_SUBJECT"; return 0; } ;;
    esac
    if git -C "$dir" rev-parse HEAD >/dev/null 2>&1; then
        git -C "$dir" log -1 --format=%s
    else
        echo "(unknown)"
    fi
}

copy_files() {
    local tg=$1
    local f
    for f in $FILES_SRC; do
        cp "$tg/src/$f" "$DEST/$f"
        say "  $f"
    done
    for f in $FILES_SOAP; do
        cp "$tg/src/soap_turbo/src/$f" "$DEST/$f"
        say "  $f"
    done
    return 0
}

report_differences() {
    local tg=$1
    local f src n=0
    for f in $FILES_SRC $FILES_SOAP; do
        case " $FILES_SRC " in
            *" $f "*) src=$tg/src/$f ;;
            *) src=$tg/src/soap_turbo/src/$f ;;
        esac
        if ! cmp -s "$src" "$DEST/$f"; then
            say "  DIFFERS  $f"
            n=$((n + 1))
        fi
    done
    if [ "$n" = 0 ]; then
        say "  every vendored file matches $tg"
        return 0
    fi
    say ""
    say "$n file(s) differ. Refresh with: tools/vendor_turbogap.sh $tg"
    return 1
}

write_provenance() {
    local tg=$1
    {
        printf '# Where these files came from\n\n'
        printf 'Written by `tools/vendor_turbogap.sh`. Do not edit these sources by\n'
        printf 'hand -- an edited copy cannot be diffed against upstream, and that is\n'
        printf 'exactly how the previous set drifted without anyone noticing.\n\n'
        printf '| | |\n|---|---|\n'
        printf '| vendored on | %s |\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '| turbogap | `%s` |\n' "$(commit_of "$tg")"
        printf '| turbogap subject | %s |\n' "$(subject_of "$tg")"
        printf '| soap_turbo | `%s` |\n' "$(commit_of "$tg/src/soap_turbo")"
        printf '| soap_turbo subject | %s |\n' "$(subject_of "$tg/src/soap_turbo")"
        printf '\n## Files\n\n'
        printf '| file | from | sha256 |\n|---|---|---|\n'
        local f where
        for f in $FILES_SRC; do
            printf '| `%s` | `src/` | `%s` |\n' "$f" "$(sha256sum "$DEST/$f" | cut -c1-16)"
        done
        for f in $FILES_SOAP; do
            printf '| `%s` | `src/soap_turbo/src/` | `%s` |\n' "$f" "$(sha256sum "$DEST/$f" | cut -c1-16)"
        done
        printf '\nCheck they are still current with:\n\n'
        printf '```sh\ntools/vendor_turbogap.sh --check /path/to/turbogap\n```\n'
    } > "$PROVENANCE"
    return 0
}

# --------------------------------------------------------------------- wiring

[ $# -eq 1 ] || die "usage: tools/vendor_turbogap.sh [--check] /path/to/turbogap"
tg=$(cd "$1" && pwd)

check_source_tree "$tg"

if [ "$check_only" = 1 ]; then
    say "comparing lib/madlib/turbogap against $tg"
    report_differences "$tg"
    exit $?
fi

say "vendoring from $tg"
copy_files "$tg"
write_provenance "$tg"
say ""
say "wrote $PROVENANCE"
