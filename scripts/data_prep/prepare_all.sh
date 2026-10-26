#!/bin/bash
# Rebuild the complete Cleopatra training data from Zenodo, ENCODE and UCSC.
#
#   bash scripts/data_prep/prepare_all.sh <data_root> [n_processes]
#
# Then set the `paths:` block of a config to <data_root> (the shipped configs
# expect data_root/ next to train.py) and run train.py. See README.md,
# "Reproducing the training data", for what each step does.

set -euo pipefail

DATA_ROOT=${1:?usage: prepare_all.sh <data_root> [n_processes]}
NPROC=${2:-8}
HERE=$(cd "$(dirname "$0")" && pwd)

bash "$HERE/00_download.sh" "$DATA_ROOT"
bash "$HERE/01_make_mcool.sh" "$DATA_ROOT" "$NPROC"
bash "$HERE/02_dump_contacts.sh" "$DATA_ROOT"
python "$HERE/03_oe_vectors.py" --data-root "$DATA_ROOT"
python "$HERE/04_epigenomics.py" --data-root "$DATA_ROOT"
python "$HERE/05_loops.py" --data-root "$DATA_ROOT"
