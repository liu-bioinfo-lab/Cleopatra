#!/bin/bash
# Step 2: dump Micro-C and RCMC contacts per chromosome as text, the format read by
# cleopatra/data.py::load_matrices.
#
#   bash scripts/data_prep/02_dump_contacts.sh <data_root>
#
# Output (8 tab-separated columns: chrom1 start1 end1 chrom2 start2 end2 count balanced):
#   <data_root>/MicroC/<cell>/<chr>_<res>bp.txt
#   <data_root>/RCMC/<cell>/<chr>_<res>bp.txt
# for chr1-22 and chrX at 500 bp and 2 kb. Resolutions missing from a file are skipped.

set -euo pipefail

DATA_ROOT=${1:?usage: 02_dump_contacts.sh <data_root>}
CHROMS=$(echo chr{1..22} chrX)

dump() {   # dump <mcool> <out_dir>
    local mcool=$1 out=$2
    mkdir -p "$out"
    for res in 500 2000; do
        if ! cooler ls "$mcool" | grep -q "/resolutions/${res}$"; then
            echo "WARNING: no ${res} bp resolution in $mcool, skipped" >&2
            continue
        fi
        for ch in $CHROMS; do
            cooler dump -r "$ch" -b --join -o "$out/${ch}_${res}bp.txt" "$mcool::/resolutions/$res"
        done
    done
}

for cell in GM12878 K562 HCT-116 H1; do
    z=${cell/HCT-116/HCT116}
    [ -e "$DATA_ROOT/raw/mcool/${cell}_microc.mcool" ] && dump "$DATA_ROOT/raw/mcool/${cell}_microc.mcool" "$DATA_ROOT/MicroC/$cell"
    dump "$DATA_ROOT/raw/zenodo/${z}_RCMC_merged.50.mcool" "$DATA_ROOT/RCMC/$cell"
done
