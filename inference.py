"""
Genome-wide contact-map inference with a trained Cleopatra model.

Usage::

    python inference.py --config configs/gm12878_500bp.yaml \\
                        --checkpoint weights/gm12878_500bp_finetune.pt \\
                        --output-dir inference/GM12878

For each cell line in the config, walks every chromosome window listed in
``HUMAN_CHR_SIZES`` and writes the predicted OE strata to
``<output-dir>/<cell>/<chr>/region<idx>_strata_oe.pkl``.
"""
from __future__ import annotations

import argparse
import os
import pickle
from time import time

import numpy as np
import torch
import yaml

from cleopatra.model import Cleopatra
from cleopatra.loader import inference_data_generator
from cleopatra.metrics import update_strata, normalize_matrix
from cleopatra.regions import HUMAN_CHR_SIZES


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _batch_to_tensors(batch_tuple):
    epis, pos_enc, distance, weights = batch_tuple
    return (
        torch.from_numpy(epis.astype(np.float32)),
        torch.from_numpy(pos_enc.astype(np.float32)),
        torch.from_numpy(distance.astype(np.int64)),
        torch.from_numpy(weights.astype(np.float32)),
    )


def _max_range(resolution):
    if resolution == 500:  return 625000
    if resolution == 2000: return 2500000
    raise ValueError(f'resolution must be 500 or 2000 (got {resolution})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True, help='Path to a Cleopatra .pt file')
    parser.add_argument('--output-dir', default='inference')
    parser.add_argument('--batch-size', type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolution = cfg['data']['resolution']
    max_range = _max_range(resolution)
    epi_names = cfg['data']['epigenetic_inputs']
    arch = cfg['model']

    model = Cleopatra(
        nBins=arch['n_bins'], nMarks=len(epi_names), verbose=1,
        positional_dim=arch['positional_dim'],
        n_distance=arch['n_bins'], distance_dim=arch['distance_dim'],
        n_SA_layers=arch['n_SA_layers'], SA_dim=arch['SA_dim'], SA_head=arch['SA_head'],
        n_Conv_layers=arch['n_Conv_layers'], Conv_dim=arch['Conv_dim'], Conv_kernel=arch['Conv_kernel'],
        n_final_layers=arch['n_final_layers'], final_layer_dim=arch['final_layer_dim'],
        final_layer_kernel=arch['final_layer_kernel'],
        insulation_loss=False, beta=0.0,
        freeze_distance_embedding=True,
        rbf_n_centers=arch['rbf_n_centers'], rbf_width_factor=arch['rbf_width_factor'],
    )
    model.load_weights(args.checkpoint, device=DEVICE)
    model = model.to(DEVICE)
    print(f'Loaded {args.checkpoint} on {DEVICE}')

    gap = max_range // 5
    norm_matrices = normalize_matrix(max_range, gap, resolution)

    for cell_line in cfg['data']['cell_lines']:
        out_root = os.path.join(args.output_dir, cell_line)
        os.makedirs(out_root, exist_ok=True)

        for region_idx, (ch, st, ed) in HUMAN_CHR_SIZES.items():
            length = (ed - st) // resolution
            strata = [np.zeros((length - i,)) for i in range(1000)]
            generator = inference_data_generator(
                {region_idx: HUMAN_CHR_SIZES[region_idx]},
                [cell_line], epi_names,
                paths={'epi_root': cfg['paths']['epi_root']},
                max_range=max_range, resolution=resolution, gap=gap,
                pos_enc_dim=arch['positional_dim'], batch_size=args.batch_size,
                n_distance=arch['n_bins'],
            )
            for _, batch_tuple, keys in generator:
                _tm = time()
                pred_np = model.predict_batch(_batch_to_tensors(batch_tuple), DEVICE)
                for j, ky in enumerate(keys):
                    _res = np.exp(pred_np[j]) - 1
                    region_start = int(ky.split('_')[-1])
                    n_th_now = (region_start - st) // gap
                    n_th_last = (ed - max_range - region_start) // gap
                    if n_th_now <= 3:
                        norm = norm_matrices[n_th_now]
                    elif n_th_last <= 3:
                        norm = norm_matrices[-n_th_last - 1]
                    else:
                        norm = norm_matrices[4]
                    strata = update_strata(strata, _res / norm, (region_start - st) // resolution)
                print(f'  batch: {time() - _tm:.1f}s')
            # Left-pad strata by st//resolution zeros so the array covers the full chromosome.
            n_zeros = st // resolution
            for i in range(1000):
                strata[i] = np.concatenate([np.zeros((n_zeros,)), strata[i]])

            out_dir = os.path.join(out_root, ch)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f'region{region_idx}_strata_oe.pkl'), 'wb') as f:
                pickle.dump(strata, f)
            print(f'  Wrote {ch} region {region_idx} ({length} bins) strata')


if __name__ == '__main__':
    main()
