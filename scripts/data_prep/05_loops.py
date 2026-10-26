"""
Step 5: per-cell loop files for loop-weighted fine-tuning and loop metrics.

Usage::

    python scripts/data_prep/05_loops.py --data-root <data_root>

Splits data/loops/all_rcmc_annotated_loops.tsv (the RCMC loop set of the
manuscript, Data S) into ``<data_root>/Loops/<cell>.txt`` with six
tab-separated columns (chrom1 start1 end1 chrom2 start2 end2), the format read
by cleopatra/data.py::load_loops. 60 HCT116 loops without a chromosome name in
the table are skipped; they were not used to train the published models either.
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOOPS = os.path.join(HERE, '..', '..', 'data', 'loops', 'all_rcmc_annotated_loops.tsv')
NAMES = {'GM12878': 'GM12878', 'H1': 'H1', 'HCT116': 'HCT-116', 'K562': 'K562'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--loops', default=LOOPS)
    a = ap.parse_args()
    df = pd.read_csv(a.loops, sep='\t')
    os.makedirs(f'{a.data_root}/Loops', exist_ok=True)
    for name, cell in NAMES.items():
        sub = df[(df.celltype == name) & df.chrom1.notna()][['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']]
        sub.to_csv(f'{a.data_root}/Loops/{cell}.txt', sep='\t', header=False, index=False)
        print(f'{cell}: {len(sub)} loops')


if __name__ == '__main__':
    main()
