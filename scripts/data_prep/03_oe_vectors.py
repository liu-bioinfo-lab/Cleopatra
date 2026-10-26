"""
Step 3: expected-contact (OE) vectors used for observed / expected normalisation.

Usage::

    python scripts/data_prep/03_oe_vectors.py --data-root <data_root>

For each genomic separation d (in bins), the vector holds the mean balanced
contact frequency over all bin pairs at separation d:

* Micro-C: one genome-wide vector per cell type and resolution, summed over
  chr1-22 and chrX and divided by the number of bin pairs at that separation
  (chromosome lengths from hg38.chrom.sizes). Distances up to 2 Mb (500 bp) or
  3 Mb (2 kb).
  -> ``<data_root>/MicroC/OE_vectors/<cell>_<res>bp.npy``
* RCMC: one vector per cell type, resolution and capture region, using only
  off-diagonal pairs with both anchors inside the captured interval below.
  Distances up to 2 Mb (500 bp) or 5 Mb (2 kb).
  -> ``<data_root>/RCMC/OE_vectors/<cell>_<res>bp_<region_idx>.npy``

Inputs are the text dumps written by 02_dump_contacts.sh. Rows without a
balanced value are ignored.
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = ['GM12878', 'H1', 'HCT-116', 'K562']
MICROC_MAX_RANGE = {500: 2000000, 2000: 3000000}
RCMC_MAX_RANGE = {500: 2000000, 2000: 5000000}

# Captured intervals of the 14 RCMC regions (hg38). Index = region index in
# cleopatra/regions.py::RCMC_REGIONS (region 1 was not used for training).
# These are the capture-probe intervals, which are slightly wider than the
# training windows in RCMC_REGIONS.
CAPTURE_REGIONS = [
    'chr6:25111952-28621868',
    'chr19:36027639-39756265',
    'chr5:157036231-160147469',
    'chr1:207626481-210338298',
    'chr4:61369236-64434279',
    'chr8:124988367-129795630',
    'chr6:29678698-32256197',
    'chrX:47081617-49442070',
    'chr1:237286814-240531042',
    'chr7:10480243-13506262',
    'chr8:62918949-66566105',
    'chr4:181289951-184014354',
    'chr3:118624552-121533938',
    'chr9:106603383-109909256',
]


def read_dump(path, chunksize=5_000_000):
    """Yield (start1, start2, balanced) arrays from a `cooler dump -b --join` text file."""
    for df in pd.read_csv(path, sep='\t', header=None, usecols=[1, 4, 7], names=['p1', 'p2', 'v'],
                          chunksize=chunksize):
        df = df.dropna()
        yield df.p1.to_numpy(), df.p2.to_numpy(), df.v.to_numpy(dtype=float)


def chrom_sizes(path=os.path.join(HERE, 'hg38.chrom.sizes')):
    sizes = {}
    for line in open(path):
        ch, size = line.split()
        sizes[ch] = int(size)
    del sizes['chrY']
    return sizes


def microc_vector(dump_dir, res):
    n = MICROC_MAX_RANGE[res] // res
    sizes = chrom_sizes()
    n_bins = sum(int(np.ceil(size / res)) for size in sizes.values())
    total = np.zeros(n)
    for ch in sizes:
        for p1, p2, v in read_dump(f'{dump_dir}/{ch}_{res}bp.txt'):
            d = np.abs(p1 - p2) // res
            keep = d < n
            total += np.bincount(d[keep], weights=v[keep], minlength=n)
    return total / (n_bins - np.arange(n))


def rcmc_vectors(dump_dir, res):
    n = RCMC_MAX_RANGE[res] // res
    out = {}
    for idx, reg in enumerate(CAPTURE_REGIONS):
        ch, coord = reg.split(':')
        st, ed = (int(x) // res * res for x in coord.split('-'))
        total = np.zeros(n)
        for p1, p2, v in read_dump(f'{dump_dir}/{ch}_{res}bp.txt'):
            lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
            d = (hi - lo) // res
            keep = (d < n) & (st <= lo) & (lo < hi) & (hi < ed)
            total += np.bincount(d[keep], weights=v[keep], minlength=n)
        out[idx] = total / ((ed - st) // res - np.arange(n))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--cells', nargs='+', default=CELLS)
    ap.add_argument('--resolutions', nargs='+', type=int, default=[500, 2000])
    ap.add_argument('--assays', nargs='+', default=['MicroC', 'RCMC'], choices=['MicroC', 'RCMC'])
    a = ap.parse_args()
    for assay in a.assays:
        os.makedirs(f'{a.data_root}/{assay}/OE_vectors', exist_ok=True)
        for cell in a.cells:
            for res in a.resolutions:
                dump_dir = f'{a.data_root}/{assay}/{cell}'
                if not os.path.exists(f'{dump_dir}/chr1_{res}bp.txt'):
                    print(f'skip {assay} {cell} {res} bp: no dumps in {dump_dir}')
                    continue
                print(f'{assay} {cell} {res} bp', flush=True)
                if assay == 'MicroC':
                    np.save(f'{a.data_root}/MicroC/OE_vectors/{cell}_{res}bp.npy', microc_vector(dump_dir, res))
                else:
                    for idx, vec in rcmc_vectors(dump_dir, res).items():
                        np.save(f'{a.data_root}/RCMC/OE_vectors/{cell}_{res}bp_{idx}.npy', vec)


if __name__ == '__main__':
    main()
