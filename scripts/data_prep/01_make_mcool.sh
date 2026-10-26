#!/bin/bash
# Step 1: build balanced multi-resolution Micro-C files.
#
#   bash scripts/data_prep/01_make_mcool.sh <data_root> [n_processes]
#
# Micro-C is distributed as single-resolution .cool files; RCMC .mcool files on
# Zenodo already contain balanced 500 bp and 2 kb resolutions and are used as is
# (do not re-balance them). Coarsening and balancing use cooler's defaults
# (iterative correction with ignore_diags=2, mad_max=5, min_nnz=10), the same
# settings as the files used to train the published models.
#
# Output: <data_root>/raw/mcool/<cell>_microc.mcool with /resolutions/500 and /resolutions/2000.
# Needs cooler >= 0.9 (pip install cooler). Balancing genome-wide 500 bp maps is
# slow and memory-hungry; run this on a compute node.

set -euo pipefail

DATA_ROOT=${1:?usage: 01_make_mcool.sh <data_root> [n_processes]}
NPROC=${2:-8}
IN=$DATA_ROOT/raw/zenodo
OUT=$DATA_ROOT/raw/mcool
mkdir -p "$OUT"

for pair in GM12878:GM12878_microc.50.cool K562:K562_microc.50.cool \
            HCT-116:HCT116_microc.50.cool H1:H1_microc.1kb.cool; do
    cell=${pair%%:*}
    src=$IN/${pair#*:}
    base=$(cooler info --field bin-size "$src")
    # zoomify can only aggregate, so a target bin size must be a multiple of the base bin size.
    res=$(for r in 500 2000; do [ $((r % base)) -eq 0 ] && echo -n "$r,"; done)
    res=${res%,}
    if [ "$res" != "500,2000" ]; then
        echo "WARNING: $src has ${base} bp bins; only ${res:-no} resolution(s) can be built for $cell" >&2
    fi
    [ -z "$res" ] && continue
    cooler zoomify --balance -p "$NPROC" -r "$res" -o "$OUT/${cell}_microc.mcool" "$src"
done
