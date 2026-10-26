"""
Step 4: bin the epigenomic bigWigs into the per-chromosome arrays read by
cleopatra/data.py::load_epigenetic_data.

Usage::

    python scripts/data_prep/04_epigenomics.py --data-root <data_root>

1. Binning: for every (cell type, assay) in epigenomic_sources.tsv, the mean
   bigWig signal in each bin (pyBigWig ``stats(exact=True)``; empty bins = 0)
   for chr1-22 and chrX. The last partial bin of a chromosome is dropped.
2. Cross-cell scaling (2 kb only, as for the published models): for every
   assay except phastCons, each cell type's track is multiplied by
   sum(GM12878 track) / sum(this cell type's track), summed over chr1-22 and
   chrX, so that all cell types have the same total signal as GM12878.
   phastCons is identical for all cell types and is not scaled. The published
   500 bp models were trained on the binned tracks without this scaling; use
   ``--scale-resolutions 500 2000`` to scale both. The factors are written to
   ``<data_root>/Epigenomics/scale_factors_<res>bp.tsv``.

Output: ``<data_root>/Epigenomics/<cell>/<chr>/<chr>_<res>bp_<assay>.npy``.
The log(x + 1) transform and the cap at 6 are applied later, when the data are
loaded for training.
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = ['GM12878', 'H1', 'HCT-116', 'K562']   # first = reference for scaling
UNSCALED = {'phastCons'}


def chrom_sizes(path=os.path.join(HERE, 'hg38.chrom.sizes')):
    sizes = {}
    for line in open(path):
        ch, size = line.split()
        sizes[ch] = int(size)
    del sizes['chrY']
    return sizes


def bin_bigwig(bw_path, res, sizes):
    import pyBigWig
    bw = pyBigWig.open(bw_path)
    out = {}
    for ch, size in sizes.items():
        n_bins = size // res
        vec = bw.stats(ch, 0, n_bins * res, exact=True, nBins=n_bins)
        out[ch] = np.array([0 if x is None else x for x in vec], dtype=float)
    bw.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--bigwig-dir', default=None, help='default: <data_root>/raw/bigwig')
    ap.add_argument('--sources', default=os.path.join(HERE, 'epigenomic_sources.tsv'))
    ap.add_argument('--resolutions', nargs='+', type=int, default=[500, 2000])
    ap.add_argument('--scale-resolutions', nargs='*', type=int, default=[2000],
                    help='resolutions at which tracks are scaled to the GM12878 total (published models: 2000 only)')
    a = ap.parse_args()
    bw_dir = a.bigwig_dir or f'{a.data_root}/raw/bigwig'
    src = pd.read_csv(a.sources, sep='\t')
    sizes = chrom_sizes()
    out_root = f'{a.data_root}/Epigenomics'
    for res in a.resolutions:
        factors = []
        for assay, rows in src.groupby('assay', sort=False):
            ref_total = None
            for cell in CELLS:
                row = rows[rows.cell == cell]
                if row.empty:
                    continue
                print(f'{res} bp  {cell}  {assay}', flush=True)
                tracks = bin_bigwig(f'{bw_dir}/{os.path.basename(row.url.iloc[0])}', res, sizes)
                total = sum(v.sum() for v in tracks.values())
                if assay in UNSCALED or res not in a.scale_resolutions:
                    scale = 1.0
                elif ref_total is None:
                    ref_total, scale = total, 1.0
                else:
                    scale = ref_total / total
                factors.append(dict(assay=assay, cell=cell, scale=scale))
                for ch, vec in tracks.items():
                    os.makedirs(f'{out_root}/{cell}/{ch}', exist_ok=True)
                    np.save(f'{out_root}/{cell}/{ch}/{ch}_{res}bp_{assay}.npy', vec * scale)
        pd.DataFrame(factors).to_csv(f'{out_root}/scale_factors_{res}bp.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
