"""
Data generators used by ``train.py``, ``evaluate.py`` and ``inference.py``.

Three generators, all sharing the same batch tuple format
``(epis, pos_encoding, distance, weights)`` consumed by ``Cleo``:

  - :func:`training_data_generator` — shuffled epoch-based generator with
    optional loop/distance weights.  Used for both PT (Micro-C) and FT (RCMC).
  - :func:`evaluation_data_generator` — deterministic single-pass generator
    that also yields ground-truth matrices and loop coordinates.
  - :func:`inference_data_generator` — genome-wide predictor generator that
    yields only the input tensors (no targets, no per-window weights).
"""
from __future__ import annotations

from time import time
import numpy as np

from .data import (
    positional_encoding,
    distance_mat,
    obtain_weight_matrix,
    load_epigenetic_data,
    load_matrices,
    load_weight_locs,
    load_weight,
)


def _init_pos_and_distance(size, pos_enc_dim, n_distance, batch_size):
    _tm = time()
    pos = positional_encoding(size, pos_enc_dim)
    pos = np.array([pos for _ in range(batch_size)])
    dist = distance_mat(size, n_distance)
    dist = np.array([dist for _ in range(batch_size)])
    t_min, t_sec = divmod(time() - _tm, 60)
    print(f'Finish generating positional encoding... {t_min:.0f}min:{t_sec:.0f}sec')
    return pos, dist


def training_data_generator(
    ch_coord, cell_lines, epi_names, skip_regions, mode,
    paths,
    max_range=1000000, resolution=500, n_distance=1250, gap=1000000,
    pos_enc_dim=8, n_epoches=100, batch_size=20,
    distance_weight=False, loop=False,
    distance_weight_rule=None, smooth=True, shuffle=True,
):
    """Yield ``(epoch, batch_idx, (epis, pos, dist, weights), targets, keys)``.

    ``paths`` is a dict with:
        - ``epi_root``: epigenomic .npy tree
        - ``contact_root``: MicroC (PT) or RCMC (FT) dump tree
        - ``oe_root``: matching OE-vector tree
        - ``loops_dir``: directory of per-cell loop annotation files
          (only needed when ``loop=True``)
    """
    size = max_range // resolution
    for idx in ch_coord:
        _, st, ed = ch_coord[idx]
        assert st % 1000 == 0 and ed % 1000 == 0

    weight_mat = obtain_weight_matrix(size, use_weight=distance_weight, rule=distance_weight_rule or {})
    pos_encoding, distance = _init_pos_and_distance(size, pos_enc_dim, n_distance, batch_size)

    all_chrs = set(ch_coord[idx][0] for idx in ch_coord)
    epi_features = load_epigenetic_data(cell_lines, all_chrs, epi_names, epi_root=paths['epi_root'], res=resolution)

    micro_mats = load_matrices(
        cell_lines, ch_coord, skip_regions, mode,
        contact_root=paths['contact_root'], oe_root=paths['oe_root'],
        length=max_range, gap=gap, resolution=resolution,
        smooth=smooth, thre=8, avg_min=1e-5, raw_thr=0.1,
    )
    weight_locs = load_weight_locs(
        mode=mode, resolution=resolution, use_loop=loop,
        all_keys=micro_mats.keys(), loops_dir=paths.get('loops_dir'), n_bins=size,
    )
    print(f'Finish processing {len(micro_mats)} contact maps')

    all_keys = list(micro_mats.keys())
    idx_list = list(range(len(all_keys)))
    n_batches = len(idx_list) // batch_size
    if n_batches == 0:
        raise RuntimeError(
            f'training_data_generator: {len(idx_list)} windows < batch_size={batch_size}. '
            'Reduce batch_size, expand ch_coord, or lower the sampling gap.'
        )
    dropped = len(idx_list) - n_batches * batch_size
    if dropped:
        print(f'WARNING: dropping {dropped} tail samples (not a multiple of batch_size).')
    np.random.seed(0)
    _tm = time()

    for _epoch in range(n_epoches):
        print(f'Epoch: {_epoch + 1}')
        if shuffle:
            np.random.shuffle(idx_list)

        for _batch in range(n_batches):
            if _epoch != 0 or _batch != 0:
                t_min, t_sec = divmod(time() - _tm, 60)
                print(f'{t_min:.0f}min:{t_sec:.0f}sec')
            _tm = time()

            print(f' Batch: {_batch + 1}')
            batch_idx = idx_list[_batch * batch_size: (_batch + 1) * batch_size]
            keys = [all_keys[k] for k in batch_idx]
            micros = np.array([micro_mats[k] for k in keys])

            epis = np.zeros((batch_size, size, len(epi_names)))
            weights = np.ones((batch_size, size, size))
            for i, key in enumerate(keys):
                cell, ch, pos = key.split('_')
                pos = int(pos)
                epis[i] = epi_features[cell][ch][pos // resolution: pos // resolution + size, :]
                info = weight_locs.get(key, {})
                weights[i] = load_weight(info, weight_mat)

            yield _epoch, _batch, (epis, pos_encoding, distance, weights), micros * weights, keys


def evaluation_data_generator(
    ch_coord, cell_lines, epi_names, skip_regions, mode,
    paths,
    max_range=1000000, resolution=500, n_distance=1250, gap=1000000,
    pos_enc_dim=8, n_epoches=1, batch_size=1, smooth=True,
    load_loops=True,
):
    """Yield ``(epoch, batch_idx, batch_tuple, targets, keys, loop_locs)``.

    ``loop_locs`` per key is a dict (possibly empty).  When
    ``load_loops`` is False, or when ``paths['loops_dir']`` is absent,
    all loop dicts are empty and downstream loop metrics are skipped.
    """
    size = max_range // resolution
    for idx in ch_coord:
        _, st, ed = ch_coord[idx]
        assert st % 1000 == 0 and ed % 1000 == 0

    pos_encoding, distance = _init_pos_and_distance(size, pos_enc_dim, n_distance, batch_size)

    all_chrs = set(ch_coord[idx][0] for idx in ch_coord)
    epi_features = load_epigenetic_data(cell_lines, all_chrs, epi_names, epi_root=paths['epi_root'], res=resolution)

    micro_mats = load_matrices(
        cell_lines, ch_coord, skip_regions, mode,
        contact_root=paths['contact_root'], oe_root=paths['oe_root'],
        length=max_range, gap=gap, resolution=resolution,
        smooth=smooth, thre=8, avg_min=1e-5, raw_thr=0.1,
    )
    use_loop = bool(load_loops and paths.get('loops_dir'))
    weight_locs = load_weight_locs(
        mode=mode, resolution=resolution, use_loop=use_loop,
        all_keys=micro_mats.keys(), loops_dir=paths.get('loops_dir'), n_bins=size,
    )
    print(f'Finish processing {len(micro_mats)} contact maps')

    all_keys = list(micro_mats.keys())
    idx_list = list(range(len(all_keys)))
    for _epoch in range(n_epoches):
        for _batch in range(len(idx_list) // batch_size):
            batch_idx = idx_list[_batch * batch_size: (_batch + 1) * batch_size]
            keys = [all_keys[k] for k in batch_idx]
            micros = np.array([micro_mats[k] for k in keys])

            epis = np.zeros((batch_size, size, len(epi_names)))
            weights = np.ones((batch_size, size, size))
            for i, key in enumerate(keys):
                cell, ch, pos = key.split('_')
                pos = int(pos)
                epis[i] = epi_features[cell][ch][pos // resolution: pos // resolution + size, :]
            loop_locs = {k: weight_locs.get(k, {}) for k in keys}

            yield _epoch, _batch, (epis, pos_encoding, distance, weights), micros, keys, loop_locs


def inference_data_generator(
    ch_coord, cell_lines, epi_names, paths,
    max_range=1000000, resolution=500, gap=1000000,
    pos_enc_dim=8, batch_size=20, n_distance=1250,
):
    """Yield ``(batch_idx, batch_tuple, keys)`` -- no targets."""
    size = max_range // resolution
    for idx in ch_coord:
        _, st, ed = ch_coord[idx]
        assert st % 1000 == 0 and ed % 1000 == 0

    weight_mat = np.ones((batch_size, size, size))
    pos_encoding, distance = _init_pos_and_distance(size, pos_enc_dim, n_distance, batch_size)

    all_chrs = set(ch_coord[idx][0] for idx in ch_coord)
    epi_features = load_epigenetic_data(cell_lines, all_chrs, epi_names, epi_root=paths['epi_root'], res=resolution)

    all_keys = []
    for cell in cell_lines:
        for idx in ch_coord:
            ch, st, ed = ch_coord[idx]
            all_keys.extend(f'{cell}_{ch}_{pos}' for pos in range(st, ed - max_range + 1, gap))

    n_batches = (len(all_keys) + batch_size - 1) // batch_size
    for _batch in range(n_batches):
        print(f' Batch: {_batch + 1}/{n_batches}')
        keys = all_keys[_batch * batch_size: (_batch + 1) * batch_size]
        epis = np.zeros((batch_size, size, len(epi_names)))
        for i, key in enumerate(keys):
            cell, ch, pos = key.split('_')
            pos = int(pos)
            epis[i] = epi_features[cell][ch][pos // resolution: pos // resolution + size, :]
        yield _batch, (epis, pos_encoding, distance, weight_mat), keys
