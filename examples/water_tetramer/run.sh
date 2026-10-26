#!/bin/bash
# (H2O)4, sobEDA + sobEDAw at BHandHLYP-D3(BJ)/6-31+G(d,p), complex basis
# (iCP=1) -- the level of template.gjf. --cart because Gaussian uses
# Cartesian d functions (6D) for the 6-31G family. Also prints the pairwise
# / many-body decomposition (not part of sobEDA.sh; --no-pairwise skips it).
cd "$(dirname "$0")" || exit 1
python ../../sobeda.py system.xyz fragment.txt \
    --xc BHANDHLYP --basis '6-31+G(d,p)' --disp d3bj --cart \
    --sobedaw 0.634 -0.065 0.702 "$@"
