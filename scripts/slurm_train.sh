#!/bin/bash
#SBATCH --job-name=cleopatra
#SBATCH --mem=128g
#SBATCH --time=72:00:00
#SBATCH --gpus=v100:1
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-v100

# Example SLURM launcher. Edit CONFIG below (and account/partition/mem/time
# for your cluster) and submit with:
#     sbatch scripts/slurm_train.sh
#
# The script runs the unified train.py entry point which handles both
# pre-training (Micro-C) and fine-tuning (RCMC) in a single job.

set -euo pipefail

CONFIG=configs/gm12878_500bp.yaml
STAGE=both

# Load your Python environment (edit for your setup).
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate cleopatra

python -u train.py --config "$CONFIG" --stage "$STAGE" >> log.txt 2>&1
