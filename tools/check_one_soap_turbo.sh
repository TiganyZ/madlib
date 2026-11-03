#!/bin/bash
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
#
# Refuse a link line that carries two soap_turbo implementations.
#
# TurboGAP's CPU and GPU soap_turbo share a module name, soap_turbo_desc, and
# therefore a symbol: __soap_turbo_desc_MOD_get_soap. Their get_soap takes 34
# arguments and 48 respectively.
#
# The linker does not object. An explicit .o beats an archive member -- the
# archive is only searched for symbols still unresolved -- so whichever is given
# directly wins, silently, and a caller compiled against the other .mod then
# makes a 34-argument call into a 48-argument routine. That is undefined
# behaviour with no diagnostic at any stage.
#
# So this is a build-time check, meant to sit in front of any link that mixes
# libmadlib.a with the device SOAP objects.
#
# usage: check_one_soap_turbo.sh <object-or-archive> [...]

set -u

SYMBOL='__soap_turbo_desc_MOD_get_soap'

usage() {
    echo "usage: $0 <object-or-archive> [...]" >&2
    exit 2
}

# Files that define the symbol, not merely reference it. nm marks a definition
# in .text with T; an undefined reference with U.
definers() {
    local f found=""
    for f in "$@"; do
        [ -e "$f" ] || continue
        if nm "$f" 2>/dev/null | grep -q " T $SYMBOL\$"; then
            found="$found $f"
        fi
    done
    echo "$found"
}

main() {
    [ $# -ge 1 ] || usage
    local found n
    found=$(definers "$@")
    n=$(echo $found | wc -w)

    if [ "$n" -le 1 ]; then
        echo "  one soap_turbo on the link line ($n definition of $SYMBOL)"
        exit 0
    fi

    echo "ERROR: $n definitions of $SYMBOL on the same link line:" >&2
    for f in $found; do echo "         $f" >&2; done
    echo >&2
    echo "       The CPU and GPU soap_turbo share a module name. Their get_soap" >&2
    echo "       takes 34 and 48 arguments. The linker will not complain -- an" >&2
    echo "       explicit object beats an archive member -- and the mismatch is" >&2
    echo "       then undefined behaviour with no diagnostic." >&2
    echo >&2
    echo "       Build one or the other, not both." >&2
    exit 1
}

main "$@"
