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
  {h1,hct116,k562}_{500,2000}bp.yaml
weights/                      # pre-trained checkpoints (see weights/README.md)
scripts/slurm_train.sh        # example SLURM launcher
train.py                      # unified PT + FT entry point
evaluate.py                   # hold-out region evaluation
inference.py                  # genome-wide prediction
scripts/data_prep/            # rebuild the training data from Zenodo / ENCODE / UCSC
data/loops/                   # RCMC loop annotations used for training
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

## Reproducing the training data

All training inputs can be rebuilt from public data with the scripts in
`scripts/data_prep/`. The shipped configs read the resulting `data_root/`
directory (next to `train.py`):

```bash
pip install cooler pyBigWig                       # in addition to requirements.txt
bash scripts/data_prep/prepare_all.sh data_root 8  # 8 = processes for cooler zoomify
python train.py --config configs/gm12878_500bp.yaml --stage both
```

`prepare_all.sh` runs six steps, which can also be run one at a time:

| Step | Script | Input | Output in `data_root/` |
| --- | --- | --- | --- |
| 0 | `00_download.sh` | Zenodo record [10.5281/zenodo.18623177](https://doi.org/10.5281/zenodo.18623177); bigWigs in `epigenomic_sources.tsv` | `raw/zenodo/`, `raw/bigwig/` |
| 1 | `01_make_mcool.sh` | Micro-C `.cool` files | `raw/mcool/<cell>_microc.mcool` (500 bp, 2 kb; balanced) |
| 2 | `02_dump_contacts.sh` | Micro-C and RCMC `.mcool` files | `MicroC/<cell>/<chr>_<res>bp.txt`, `RCMC/<cell>/<chr>_<res>bp.txt` |
| 3 | `03_oe_vectors.py` | contact dumps | `MicroC/OE_vectors/<cell>_<res>bp.npy`, `RCMC/OE_vectors/<cell>_<res>bp_<region_idx>.npy` |
| 4 | `04_epigenomics.py` | bigWigs | `Epigenomics/<cell>/<chr>/<chr>_<res>bp_<mark>.npy` |
| 5 | `05_loops.py` | `data/loops/all_rcmc_annotated_loops.tsv` | `Loops/<cell>.txt` |

Resulting layout:

```
data_root/
├── MicroC/
│   ├── <cell>/<chr>_<res>bp.txt          # cooler dump output
│   └── OE_vectors/<cell>_<res>bp.npy     # genome-wide expected vector
├── RCMC/
│   ├── <cell>/<chr>_<res>bp.txt
│   └── OE_vectors/<cell>_<res>bp_<region_idx>.npy
├── Epigenomics/
│   └── <cell>/<chr>/<chr>_<res>bp_<mark>.npy
└── Loops/<cell>.txt
```

Cell-type keys are `GM12878`, `H1`, `HCT-116` and `K562`; chromosomes are
chr1-22 and chrX; `<res>` is 500 or 2000.

### Contact maps (steps 1-2)

Micro-C is distributed as single-resolution `.cool` files. Step 1 builds the
500 bp and 2 kb resolutions and balances them with cooler's default iterative
correction (`cooler zoomify --balance`). The RCMC `.mcool` files already contain
balanced resolutions and are used as distributed. Step 2 writes one text file
per chromosome with `cooler dump -r <chr> -b --join` (8 columns: chrom1 start1
end1 chrom2 start2 end2 count balanced). Training uses the balanced column.

### Expected (OE) vectors (step 3)

Each vector holds, for every genomic separation, the mean balanced contact
frequency over all bin pairs at that separation. Contact maps are divided by
it (observed / expected) before the log(x + 1) transform.

- **Micro-C**: one genome-wide vector per cell type and resolution (chr1-22,
  chrX; separations up to 2 Mb at 500 bp, 3 Mb at 2 kb).
- **RCMC**: one vector per cell type, resolution and capture region, from
  off-diagonal pairs with both anchors inside the captured interval
  (separations up to 2 Mb at 500 bp, 5 Mb at 2 kb). The captured intervals are
  listed in `03_oe_vectors.py`; region indices match
  `cleopatra/regions.py::RCMC_REGIONS`.

### Epigenomic tracks (step 4)

The 19 inputs are 18 ENCODE4 signal p-value bigWigs per cell type (hg38;
accessions in `scripts/data_prep/epigenomic_sources.tsv` and the manuscript's
Data S table) and the UCSC hg38 phastCons100way track. Each bigWig is averaged
in 500 bp or 2 kb bins with `pyBigWig` (empty bins = 0). At 2 kb, every track
except phastCons is then scaled so that its genome-wide total equals that of
GM12878; the published 500 bp models were trained on unscaled tracks, and the
script reproduces both (`--scale-resolutions` changes this). The log(x + 1)
transform and the cap at 6 are applied when the data are loaded for training.

### Loop annotations (step 5)

`data/loops/all_rcmc_annotated_loops.tsv` is the RCMC loop set of the
manuscript (manual annotations and CHIRON calls; Data S table). Step 5 splits
it into one headerless file per cell type with six columns
(`chrom1 start1 end1 chrom2 start2 end2`). Loops up-weight the fine-tuning
loss around loop pixels (`finetune.loop_weight`, `finetune.loop_mode`) and are
used by `evaluate.py` for loop-strength correlations.

### `ExampleData/`

`ExampleData/` shows the file formats on a few small files (it is not a
complete dataset); `ExampleData/Structures/Loops/` contains the full loop
files written by step 5.

---

## Configuration

Every training / inference run is driven by a YAML config. `configs/`
has one config per cell type and resolution with the settings of the
published models; their `paths:` block points at `data_root/` (see
"Reproducing the training data"). Edit `paths:` if your data live
elsewhere. All model hyperparameters are exposed in the `model:` block so
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
| `finetune`      | `loop_weight`                | Enable loop-pixel upweighting (requires `paths.loops_dir`) |
| `finetune`      | `loop_mode`                  | `compound` (overlapping loop squares multiply; 500 bp) or `fixed` (each pixel up-weighted once; 2 kb) |
| `pretrain`/`finetune` | `smooth_inputs`        | 5×5 smoothing of the contact-map targets (500 bp: true; 2 kb: false) |
| `finetune`      | `freeze_distance_embedding`  | Freeze RBF projection during FT (canonical: true)          |

---

## Pre-trained model

`weights/gm12878_500bp_finetune.pt` is the canonical GM12878 500 bp
Cleopatra checkpoint (see `weights/README.md` for load code).

Additional cell-type / resolution checkpoints and the full set of
genome-wide predictions are hosted on Zenodo:

- Predictions: **10.5281/zenodo.20739414**
- Source RCMC / Micro-C data: **10.5281/zenodo.18623177**

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
