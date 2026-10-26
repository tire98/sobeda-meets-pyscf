#!/bin/bash
# Same dimer and basis with wB97M-V (not available for sobEDA.sh): five-term
# sobEDA with a combined E_xc; no D3 (VV10 is built in), no sobEDAw,
# no Gaussian reference.
cd "$(dirname "$0")" || exit 1
python ../../sobeda.py system.xyz fragment.txt \
    --xc WB97M-V --basis '6-311+G(2d,p)' --outdir output_wb97mv "$@"
