# Cleopatra

**Cleopatra** is an attention-based deep-learning model that predicts
ultra-high-resolution 3D genome contact maps (500 bp and 2 kb bin sizes)
from one-dimensional epigenomic inputs. It is trained with a two-stage
strategy: **pre-train on genome-wide Micro-C, then fine-tune on
Region-Capture Micro-C (RCMC)**.

This repository contains the reference PyTorch implementation, the
canonical GM12878 500 bp checkpoint, and unified `train.py` /
`evaluate.py` / `inference.py` entry points driven by YAML configs.

If you use Cleopatra, please cite:

> Hong C.K.Y., Feng F., Ramanathan V., Liu J., Hansen A.S.
> *Cleopatra: attention-based prediction of ultra-high-resolution 3D
> genome architecture from epigenomic inputs.* (2026)

---

## Repository layout

```
cleopatra/                    # Python package
  model.py                    # Cleopatra model (DenseNet-style + SA + RBF distance encoding)
  loader.py                   # Training / evaluation / inference data generators
  data.py                     # Contact map + epigenomics loading, weight construction
  metrics.py                  # Pearson / SSIM / PSNR / SCC / loop / insulation metrics
  visualize.py                # Contact-map + epigenomics track plots
  regions.py                  # Chromosome & RCMC region catalogs
configs/                      # YAML configs (one per cell type × resolution)
  gm12878_500bp.yaml          # canonical published config
  gm12878_2000bp.yaml
weights/                      # pre-trained checkpoints (see weights/README.md)
scripts/slurm_train.sh        # example SLURM launcher
train.py                      # unified PT + FT entry point
evaluate.py                   # hold-out region evaluation
inference.py                  # genome-wide prediction
ExampleData/                  # small example files showing the expected data layout
requirements.txt
```

---

## Requirements

Python 3.9+ with:

- PyTorch ≥ 2.0 (CUDA build recommended)
- NumPy, SciPy, pandas, PyYAML, scikit-image, matplotlib

Install:

```bash
git clone https://github.com/liu-bioinfo-lab/Cleopatra.git
cd Cleopatra
pip install -e .            # installs the `cleopatra` package and deps
```

(A plain `pip install -r requirements.txt` also works if you plan to run
the scripts from the repo root without installing the package.)

Training a full 500 bp / 200-epoch fine-tune uses ~64 GB RAM (contact
maps are held in memory) and ~12 h on a single V100 or A100.

---

## Quick start

```bash
# Pre-train + fine-tune from a single config
python train.py --config configs/gm12878_500bp.yaml --stage both

# Only fine-tune, reusing an existing PT checkpoint
python train.py --config configs/gm12878_500bp.yaml --stage finetune \
                --pretrain-checkpoint pretrain_60.pt

# Hold-out RCMC evaluation
python evaluate.py --config configs/gm12878_500bp.yaml \
                   --checkpoint weights/gm12878_500bp_finetune.pt \
                   --output-dir evaluation_GM12878

# Genome-wide inference
python inference.py --config configs/gm12878_500bp.yaml \
                    --checkpoint weights/gm12878_500bp_finetune.pt \
                    --output-dir inference_GM12878
```

If `--stage both` finds an existing `pretrain_60.pt` next to `train.py`,
it skips the (expensive) pre-training stage and jumps straight to
fine-tuning. This is the typical iteration workflow.

Pass `--seed <int>` to any `train.py` run for a deterministic
Python/NumPy/Torch/cuDNN seed. Omit it for stochastic runs (the
manuscript models were trained without a fixed seed).

---

## Data preparation

Cleopatra accepts **19 one-dimensional genomic tracks** per cell type
(chromatin accessibility, histone marks, transcription-factor ChIP-seq,
and phastCons conservation — see the config file for the exact list).
Contact-map targets are Micro-C for pre-training and RCMC for
fine-tuning.

> **Note on `ExampleData/`.** The bundled `ExampleData/` directory is a
> **schema demo**, not a runnable dataset: it contains one loop file
> per cell line, GM12878 OE vectors, one HCT-116 contact-map dump and
> one HCT-116 H3K27ac track. Running the shipped config out of the box
> will fail with missing files. Point `paths.*` at a full data root
> (see Zenodo IDs below) before running.

Expected on-disk layout (matches the schema shown in `ExampleData/`):

```
<data_root>/
├── MicroC/
│   ├── <cell>/<chr>_<res>bp.txt          # cooler dump output
│   └── OE_vectors/<cell>_<res>bp.npy     # genome-wide OE vector
├── RCMC/
│   ├── <cell>/<chr>_<res>bp.txt
│   └── OE_vectors/<cell>_<res>bp_<region_idx>.npy
├── Epigenomics/
│   └── <cell>/<chr>/<chr>_<res>bp_<mark>.npy
└── Structures/
    └── Loops/all_rcmc_annotated_loops.tsv
```

### Contact-map files

Each contact-map text file is produced by
[`cooler dump`](https://cooler.readthedocs.io/en/latest/cli.html#cooler-dump):

```bash
cooler dump -r "chr${CHR}" -b --join -o "${OUT_TXT}" "${MCOOL_FILE}::/resolutions/${RES}"
```

### OE vectors

Store the average contact frequency at each genomic separation, used for
observed / expected normalisation.

- **Micro-C** — one genome-wide vector per (cell type, resolution):
  `<cell>_<res>bp.npy`.
- **RCMC** — one vector per (cell type, resolution, RCMC region):
  `<cell>_<res>bp_<region_idx>.npy`. Region indices match
  `cleopatra/regions.py::RCMC_REGIONS`.

### Epigenomic tracks

Each track is a per-chromosome `.npy` array generated from a bigWig via
[`pyBigWig`](https://github.com/deeptools/pyBigWig). Refer to the
manuscript's supplementary table for ENCODE bigWig accession IDs.

### Loop annotations (optional)

`paths.loops_dir` points at a directory containing one file per cell
line, `<cell>.txt`. Each file is headerless and whitespace-separated
with six columns:

```
chrom1  start1  end1  chrom2  start2  end2
```

Chromosome names may be given bare (`5`) or with a `chr` prefix
(`chr5`); a `chr` prefix is added if missing. Loop annotations are
used when `finetune.loop_weight: true` to upweight loop pixels during
training, and by `evaluate.py` when computing loop-intensity Pearson.
Both usages are optional — with `loops_dir` unset, loop weights are
skipped in FT and loop metrics are omitted from evaluation.

---

## Configuration

Every training / inference run is driven by a YAML config. Copy
`configs/gm12878_500bp.yaml`, edit the `paths:` block to point at your
data root, and change `data.cell_lines` / `data.resolution` for other
targets. All model hyperparameters are exposed in the `model:` block so
that new ablations can be run without touching Python.

Key knobs:

| Section         | Field                        | Meaning                                       |
| ---             | ---                          | ---                                           |
| `data`          | `resolution`                 | 500 or 2000 bp                                |
| `data`          | `cell_lines`                 | One or more cell-type keys                    |
| `model`         | `positional_dim`, `distance_dim` | Positional / distance embedding sizes     |
| `model`         | `SA_dim`, `SA_head`          | Self-attention channel width and head count   |
| `model`         | `rbf_width_factor`           | RBF kernel width in units of centre spacing   |
| `pretrain`/`finetune` | `n_epochs`, `learning_rate`, `gap` | Stage-specific training schedule      |
| `finetune`      | `training_regions`           | RCMC region indices used for training (holdouts: 3, 5, 10) |
| `finetune`      | `loop_weight`                | Enable loop-pixel upweighting (requires `paths.loops_tsv`) |
| `finetune`      | `freeze_distance_embedding`  | Freeze RBF projection during FT (canonical: true)          |

---

## Pre-trained model

`weights/gm12878_500bp_finetune.pt` is the canonical GM12878 500 bp
Cleopatra checkpoint (see `weights/README.md` for load code).

Additional cell-type / resolution checkpoints and the full set of
genome-wide predictions are hosted on Zenodo:

- Predictions: **10.5281/zenodo.15299993**
- Source RCMC / Micro-C data: **10.5281/zenodo.15303879**

Interactive contact-map visualisations for all four cell lines are also
available via HiGlass (see the manuscript).

---

## Repositories

- Model code: <https://github.com/liu-bioinfo-lab/Cleopatra>
- Analysis code (figures & downstream): <https://github.com/ahansenlab/cleopatra_analysis_code>
- CHIRON loop caller: <https://github.com/ahansenlab/chiron>

---

## Contact

For issues or contributions, please open a GitHub issue or email
`fanfeng@umich.edu`.
