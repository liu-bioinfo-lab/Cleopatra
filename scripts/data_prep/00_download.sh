#!/bin/bash
# Step 0: download the raw inputs needed to rebuild the Cleopatra training data.
#
#   bash scripts/data_prep/00_download.sh <data_root>
#
# Downloads into <data_root>/raw/:
#   zenodo/   Micro-C (.cool) and RCMC (.mcool) contact maps from Zenodo
#             (doi:10.5281/zenodo.18623177)
#   bigwig/   the 18 ENCODE4 signal p-value bigWigs per cell type and the UCSC
#             hg38 phastCons100way bigWig, as listed in epigenomic_sources.tsv
#
# The Zenodo files are ~54 GB in total (Micro-C ~46 GB, RCMC ~8 GB).
# wget -c resumes interrupted downloads, so the script can simply be re-run.

set -euo pipefail

DATA_ROOT=${1:?usage: 00_download.sh <data_root>}
HERE=$(cd "$(dirname "$0")" && pwd)
ZENODO=https://zenodo.org/records/18623177/files
RAW=$DATA_ROOT/raw
mkdir -p "$RAW/zenodo" "$RAW/bigwig"

# Contact maps (file names as on Zenodo).
for f in \
    GM12878_microc.50.cool K562_microc.50.cool HCT116_microc.50.cool H1_microc.1kb.cool \
    GM12878_RCMC_merged.50.mcool K562_RCMC_merged.50.mcool HCT116_RCMC_merged.50.mcool H1_RCMC_merged.50.mcool \
    RCMC_regions_overview.xlsx; do
    wget -c -q --show-progress -O "$RAW/zenodo/$f" "$ZENODO/$f?download=1"
done

# Epigenomic tracks. Each URL is downloaded once and stored under its own file
# name (the phastCons bigWig is shared by all four cell types).
tail -n +2 "$HERE/epigenomic_sources.tsv" | cut -f4 | sort -u | while read -r url; do
    wget -c -q --show-progress -O "$RAW/bigwig/$(basename "$url")" "$url"
done
