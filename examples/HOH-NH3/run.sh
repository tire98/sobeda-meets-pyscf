#!/bin/bash
# H2O...NH3, sobEDA + sobEDAw at B3LYP-D3(BJ)/6-311+G(2d,p), complex basis
# (iCP=1) -- the level of template.gjf. c, a, r are the sobEDA.sh values for
# this level. Compare the output with reference_result.txt.
# Extra options are passed through, e.g.  ./run.sh --threads 8
cd "$(dirname "$0")" || exit 1
python ../../sobeda.py system.xyz fragment.txt \
    --xc B3LYP --basis '6-311+G(2d,p)' --disp d3bj \
    --sobedaw 0.550 0.037 2.750 "$@"
