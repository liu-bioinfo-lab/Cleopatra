"""
Data utilities for Cleopatra.

Paths are provided by the caller (via a config YAML consumed by
`train.py` / `inference.py`) rather than baked into the module, so the
same code can be pointed at any cell type / resolution / directory layout.

Expected on-disk layout (matches ``ExampleData/``)::

    <data_root>/
        MicroC/
            <cell>/                        # e.g. GM12878/, HCT-116/, ...
                <chr>_<resolution>bp.txt   # `cooler dump` output
            OE_vectors/
                <cell>_<resolution>bp.npy  # genome-wide OE vector
        RCMC/
            <cell>/
                <chr>_<resolution>bp.txt
            OE_vectors/
                <cell>_<resolution>bp_<region_idx>.npy
        Epigenomics/
            <cell>/
                <chr>/
                    <chr>_<resolution>bp_<mark>.npy
        Structures/
            Loops/
                <cell>.txt                 # optional; see load_loops()
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.signal import convolve2d


def obtain_weight_matrix(size, use_weight, rule):
    mat = np.ones((size, size))
    if use_weight:
        w_1d = np.ones((size,))
        for i in range(rule['start_loc'], rule['end_loc']):
            w_1d[i] = rule['start_weight'] + (i - rule['start_loc']) / (
                rule['end_loc'] - rule['start_loc']
            ) * (rule['end_weight'] - rule['start_weight'])
        for i in range(size):
            for j in range(size):
                mat[i, j] = w_1d[abs(i - j)]
    return mat


def positional_encoding(length=1250, position_dim=8):
    assert position_dim % 2 == 0
    position = np.zeros((length, position_dim))
    for i in range(position_dim // 2):
        position[:, 2 * i] = np.sin(
            [pp / (10000 ** (2 * i / position_dim)) for pp in range(length)]
        )
        position[:, 2 * i + 1] = np.cos(
            [pp / (10000 ** (2 * i / position_dim)) for pp in range(length)]
        )
    return position


def distance_mat(length=1250, max_distance=800):
    m1 = np.tile(np.arange(length), (length, 1))
    m2 = m1.T
    dis = np.abs(m1 - m2)
    dis[dis > max_distance] = max_distance
    return dis


def load_loops(loops_dir, cell_lines=('GM12878', 'H1', 'HCT-116', 'K562')):
    """Load loop annotations from ``loops_dir/<cell>.txt``.

    Each file is a headerless whitespace-separated table with six columns
    ``chrom1  start1  end1  chrom2  start2  end2``.  Chromosome names may
    be given either bare (``5``) or prefixed (``chr5``); a ``chr`` prefix
    is added if missing.

    Returns a dict ``{cell: {chr: sorted (N, 2) int array of anchor
    midpoints}}``.
    """
    all_loops = {}
    for cell in cell_lines:
        path = os.path.join(loops_dir, f'{cell}.txt')
        if not os.path.exists(path):
            continue
        cell_dict = {}
        with open(path) as f:
            for line in f:
                lst = line.strip().split()
                if len(lst) < 6:
                    continue
                ch = lst[0] if str(lst[0]).startswith('chr') else f'chr{lst[0]}'
                s1, e1, s2, e2 = int(lst[1]), int(lst[2]), int(lst[4]), int(lst[5])
                p1, p2 = (s1 + e1) // 2, (s2 + e2) // 2
                if p1 > p2:
                    p1, p2 = p2, p1
                cell_dict.setdefault(ch, []).append((p1, p2))
        all_loops[cell] = {
            ch: np.array(sorted(v), dtype=int) for ch, v in cell_dict.items()
        }
    return all_loops


def load_weight_locs(mode, resolution, use_loop, all_keys, loops_dir=None, n_bins=1250):
    """Map each training-window key to a dict of feature locations used for
    weight-map construction.  Only ``loop`` weights are supported in this
    release."""
    key2weight_loc = {}
    if not use_loop:
        return key2weight_loc
    if loops_dir is None:
        raise ValueError('use_loop=True requires paths.loops_dir in the config')
    loop_list = load_loops(loops_dir)
    for key in all_keys:
        cell, ch, st = key.split('_')
        st, ed = int(st), int(st) + resolution * n_bins
        if cell in loop_list and ch in loop_list[cell]:
            idx_st = np.searchsorted(loop_list[cell][ch][:, 0], st, side='right')
            idx_ed = np.searchsorted(loop_list[cell][ch][:, 0], ed, side='right')
            if idx_st == idx_ed:
                continue
            all_locs = (loop_list[cell][ch][idx_st:idx_ed, :] - st) // resolution
            key2weight_loc.setdefault(key, {})['loop'] = all_locs
    return key2weight_loc


def load_weight(weight_info_dict, original_weight, loop_radius=10, loop_weight=2.5, loop_mode='compound'):
    """loop_mode='compound': overlapping loop squares multiply (500 bp models);
    loop_mode='fixed': each pixel is up-weighted at most once (2 kb models)."""
    weight = np.copy(original_weight)
    n_bins = original_weight.shape[0]
    if 'loop' in weight_info_dict and loop_mode == 'fixed':
        fold = np.ones(original_weight.shape)
        for (x1, y1) in weight_info_dict['loop']:
            if (
                x1 - loop_radius >= 0 and y1 - loop_radius >= 0
                and x1 + loop_radius < n_bins and y1 + loop_radius < n_bins
            ):
                fold[x1 - loop_radius:x1 + loop_radius + 1, y1 - loop_radius:y1 + loop_radius + 1] = loop_weight
                fold[y1 - loop_radius:y1 + loop_radius + 1, x1 - loop_radius:x1 + loop_radius + 1] = loop_weight
        return weight * fold
    if 'loop' in weight_info_dict:
        for (x1, y1) in weight_info_dict['loop']:
            if (
                x1 - loop_radius >= 0 and y1 - loop_radius >= 0
                and x1 + loop_radius < n_bins and y1 + loop_radius < n_bins
            ):
                weight[
                    x1 - loop_radius:x1 + loop_radius + 1,
                    y1 - loop_radius:y1 + loop_radius + 1
                ] *= loop_weight
                weight[
                    y1 - loop_radius:y1 + loop_radius + 1,
                    x1 - loop_radius:x1 + loop_radius + 1
                ] *= loop_weight
    return weight


def load_epigenetic_data(cell_lines, chromosomes, epi_names, epi_root, res=500, verbose=1):
    """Load per-chromosome epigenomic tracks from
    ``<epi_root>/<cell>/<chr>/<chr>_<res>bp_<mark>.npy``."""
    epigenetic_data = {}
    for cell_line in cell_lines:
        epi_data = {}
        for ch in chromosomes:
            epi_data[ch] = None
            for i, k in enumerate(epi_names):
                src = os.path.join(
                    epi_root, cell_line, ch, f'{ch}_{res}bp_{k}.npy'
                )
                s = np.load(src)
                s = np.log(s + 1)
                s[s > 6] = 6
                if verbose:
                    print(' Loading epigenomics...', ch, k, len(s))
                if i == 0:
                    epi_data[ch] = np.zeros((len(s), len(epi_names)))
                correct_size = len(epi_data[ch])
                if len(s) >= correct_size:
                    epi_data[ch][:, i] = s[:correct_size]
                else:
                    epi_data[ch][:len(s), i] = s
        epigenetic_data[cell_line] = epi_data
    return epigenetic_data


def load_matrices(
    cell_lines, ch_coord, skip_regions, mode,
    contact_root, oe_root,
    length=2000000, gap=2000000, resolution=500,
    smooth=True, thre=8, avg_min=1e-5, raw_thr=0.1,
):
    """Load contact matrices from ``cooler dump`` text files and O/E-normalise.

    Args:
        contact_root: directory containing ``<cell>/<chr>_<res>bp.txt`` files.
        oe_root: directory containing OE vectors.
            - Pre-training (``mode='PT'``): ``<cell>_<res>bp.npy``.
            - Fine-tuning (``mode='FT'``): ``<cell>_<res>bp_<region_idx>.npy``.
    """
    if mode not in ('PT', 'FT'):
        raise ValueError(f'mode {mode!r} is not supported (use "PT" or "FT")')
    mat_size = length // resolution

    print('Loading contact maps...')
    keys = []
    for idx in ch_coord:
        ch, st, ed = ch_coord[idx]
        keys.extend([f'{ch}_{pos}' for pos in range(st, ed - length + 1, gap)])
    n_mats = len(cell_lines) * len(keys)
    mem_gb = n_mats * mat_size ** 2 * 8 / 1024 ** 3
    print(f' ... {n_mats} matrices will be stored in memory (~{mem_gb:.1f} GB)')

    all_matrices = {}
    for cell in cell_lines:
        for key in keys:
            all_matrices[f'{cell}_{key}'] = np.zeros((mat_size, mat_size))
        for idx in ch_coord:
            if mode == 'PT':
                norm_path = os.path.join(oe_root, f'{cell}_{resolution}bp.npy')
            else:
                norm_path = os.path.join(oe_root, f'{cell}_{resolution}bp_{idx}.npy')
            norm_array = np.load(norm_path)
            norm_array[norm_array < avg_min] = avg_min

            ch, st, ed = ch_coord[idx]
            contact_file = os.path.join(contact_root, cell, f'{ch}_{resolution}bp.txt')
            print(f'   ... {ch}:{st}-{ed}  ({contact_file})')
            with open(contact_file) as fh:
                for line in fh:
                    lst = line.strip().split()
                    if len(lst) != 8:
                        continue
                    p1, p2 = int(lst[1]), int(lst[4])
                    v0, v = float(lst[-2]), float(lst[-1])
                    if abs(p1 - p2) >= length or v0 < raw_thr:
                        continue
                    p1_bins, p2_bins = set(), set()
                    p1_bin, p2_bin = (p1 - st) // gap, (p2 - st) // gap
                    while p1_bin >= 0:
                        if st + p1_bin * gap + length <= p1:
                            break
                        elif st + p1_bin * gap <= p1 < st + p1_bin * gap + length <= ed:
                            p1_bins.add(p1_bin)
                        p1_bin -= 1
                    while p2_bin >= 0:
                        if st + p2_bin * gap + length <= p2:
                            break
                        elif st + p2_bin * gap <= p2 < st + p2_bin * gap + length <= ed:
                            p2_bins.add(p2_bin)
                        p2_bin -= 1
                    for bin_idx in p1_bins.intersection(p2_bins):
                        mat_st = st + bin_idx * gap
                        pp1, pp2 = (p1 - mat_st) // resolution, (p2 - mat_st) // resolution
                        key = f'{cell}_{ch}_{mat_st}'
                        all_matrices[key][pp1, pp2] += v / norm_array[abs(pp1 - pp2)]
                        if pp1 != pp2:
                            all_matrices[key][pp2, pp1] += v / norm_array[abs(pp1 - pp2)]

    # Drop windows that fall entirely inside a skip region first, so we
    # don't waste smoothing work on matrices we're about to delete.
    to_del = []
    for key in all_matrices:
        cell, ch, mat_st = key.split('_')
        mat_st = int(mat_st)
        mat_ed = mat_st + length
        for skip_chr, skip_st, skip_ed in skip_regions.values():
            if ch == skip_chr and mat_st >= skip_st and mat_ed <= skip_ed:
                to_del.append(key)
                break
    for key in to_del:
        del all_matrices[key]

    for key in all_matrices:
        if smooth:
            all_matrices[key] = convolve2d(
                np.log(all_matrices[key] + 1), np.ones((5, 5)) / 25, mode='same'
            )
        else:
            all_matrices[key] = np.log(all_matrices[key] + 1)
        all_matrices[key][all_matrices[key] > thre] = thre
    return all_matrices
