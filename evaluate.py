"""
Hold-out evaluation for a trained Cleopatra model.

Usage::

    python evaluate.py --config configs/gm12878_500bp.yaml \\
                       --checkpoint finetune_200.pt \\
                       --output-dir evaluation_GM12878

Writes patch-level and region-level metrics against every RCMC region
listed in ``RCMC_REGIONS`` (both training and hold-out).  Metrics include
Pearson / Spearman / MSE / SSIM / PSNR per 500 kb patch, loop-intensity
correlations, multi-scale insulation correlations, and stratum-adjusted
correlation (SCC).
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import yaml

from cleopatra.model import Cleopatra
from cleopatra.loader import evaluation_data_generator
from cleopatra.metrics import (
    correlation_per_patch, mse_per_patch, psnr_per_patch, ssim_per_patch,
    compare_loop_intensities, multiscale_insulation, normalize_matrix,
    update_strata, calculate_scc, calculate_loop_correlation,
    calculate_insulation_correlation,
)
from cleopatra.regions import RCMC_REGIONS


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
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', default='evaluation')
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
        insulation_loss=False, beta=0.0, freeze_distance_embedding=True,
        rbf_n_centers=arch['rbf_n_centers'], rbf_width_factor=arch['rbf_width_factor'],
    )
    model.load_weights(args.checkpoint, device=DEVICE)
    model = model.to(DEVICE)

    os.makedirs(args.output_dir, exist_ok=True)
    norm_matrices = normalize_matrix(max_range, max_range // 5, resolution)
    gap = max_range // 5
    n_bins = arch['n_bins']
    # Evaluation crops the outer 10% of each side (the same padding used in the
    # RCMC sliding-window setup) so that per-patch metrics only cover the
    # region where the sliding window has full coverage.
    crop_pad = n_bins // 10
    crop_slice = slice(crop_pad, n_bins - crop_pad)

    for cell_line in cfg['data']['cell_lines']:
        out_dir = os.path.join(args.output_dir, cell_line)
        os.makedirs(out_dir, exist_ok=True)
        correlation_table = []
        loop_int = {'true': [], 'pred': []}
        pseudoloop_int = {'true': [], 'pred': []}
        insulation_scores = {}
        insulation_windows = [20, 100, 200]
        scc = {}

        for region_idx, (ch, st, ed) in RCMC_REGIONS.items():
            length = (ed - st) // resolution
            insulation_scores[region_idx] = {
                'true': np.zeros((len(insulation_windows), length)),
                'pred': np.zeros((len(insulation_windows), length)),
            }
            ins_counts = np.zeros((len(insulation_windows), length)) + 1e-8
            strata = {'true': [np.zeros((length - i,)) for i in range(1000)],
                      'pred': [np.zeros((length - i,)) for i in range(1000)]}

            generator = evaluation_data_generator(
                {region_idx: RCMC_REGIONS[region_idx]}, [cell_line], epi_names, {}, mode='FT',
                paths={
                    'epi_root':     cfg['paths']['epi_root'],
                    'contact_root': cfg['paths']['finetune_contact_root'],
                    'oe_root':      cfg['paths']['finetune_oe_root'],
                    'loops_dir':    cfg['paths'].get('loops_dir'),
                },
                max_range=max_range, resolution=resolution,
                n_distance=arch['n_bins'], gap=gap,
                pos_enc_dim=arch['positional_dim'], n_epoches=1, batch_size=1,
                smooth=cfg['finetune'].get('smooth_inputs', True),
            )
            for _, _, batch_tuple, targets, keys, loop_locs in generator:
                pred = model.predict_batch(_batch_to_tensors(batch_tuple), DEVICE)
                _res, _tgt, ky = pred[0], targets[0], keys[0]
                region_start = int(ky.split('_')[-1])
                loops = loop_locs[ky].get('loop', [])

                t_crop = _tgt[crop_slice, crop_slice]
                p_crop = _res[crop_slice, crop_slice]
                spearmanR, pearsonR = correlation_per_patch(t_crop, p_crop)
                correlation_table.append([
                    ky, spearmanR, pearsonR,
                    mse_per_patch(t_crop, p_crop),
                    psnr_per_patch(t_crop, p_crop),
                    ssim_per_patch(t_crop, p_crop),
                ])
                if len(loops) > 0:
                    lt, lp, plt_, plp = compare_loop_intensities(_tgt, _res, loops)
                    loop_int['true'].extend(lt);   loop_int['pred'].extend(lp)
                    pseudoloop_int['true'].extend(plt_); pseudoloop_int['pred'].extend(plp)

                ins_true, ins_pred, ins_ct = multiscale_insulation(_tgt, _res, window_sizes=insulation_windows)
                idx_st = (region_start - st) // resolution
                insulation_scores[region_idx]['true'][:, idx_st:idx_st + n_bins] += ins_true
                insulation_scores[region_idx]['pred'][:, idx_st:idx_st + n_bins] += ins_pred
                ins_counts[:, idx_st:idx_st + n_bins] += ins_ct

                n_th_now = (region_start - st) // gap
                n_th_last = (ed - max_range - region_start) // gap
                if n_th_now <= 3:
                    norm = norm_matrices[n_th_now]
                elif n_th_last <= 3:
                    norm = norm_matrices[-n_th_last - 1]
                else:
                    norm = norm_matrices[4]
                strata['pred'] = update_strata(strata['pred'], _res / norm, idx_st)
                strata['true'] = update_strata(strata['true'], _tgt / norm, idx_st)

            scc[region_idx] = calculate_scc(strata)
            insulation_scores[region_idx]['true'] /= ins_counts
            insulation_scores[region_idx]['pred'] /= ins_counts

        # Write outputs
        with open(os.path.join(out_dir, 'loop_intensities.pkl'), 'wb') as f:
            pickle.dump(loop_int, f)
        with open(os.path.join(out_dir, 'pseudoloop_intensities.pkl'), 'wb') as f:
            pickle.dump(pseudoloop_int, f)
        with open(os.path.join(out_dir, 'insulation_scores.pkl'), 'wb') as f:
            pickle.dump(insulation_scores, f)

        loop_sR, loop_pR = calculate_loop_correlation(loop_int)
        pseudo_sR, pseudo_pR = calculate_loop_correlation(pseudoloop_int)
        pd.DataFrame([
            ['LoopIntensitiesSpearmanR', loop_sR],
            ['LoopIntensitiesPearsonR', loop_pR],
            ['PseudoLoopIntensitiesSpearmanR', pseudo_sR],
            ['PseudoLoopIntensitiesPearsonR', pseudo_pR],
        ]).to_csv(os.path.join(out_dir, 'loop_intensities_correlations.csv'), index=False)

        pd.DataFrame(
            calculate_insulation_correlation(insulation_scores),
            columns=['Region'] + [f'Window_{w}' for w in insulation_windows],
        ).to_csv(os.path.join(out_dir, 'insulation_scores_correlations.csv'), index=False)

        pd.DataFrame(scc).to_csv(os.path.join(out_dir, 'scc_correlations.csv'), index=True)

        df_perRegion = pd.DataFrame(
            correlation_table,
            columns=['Region', 'SpearmanR', 'PearsonR', 'MSE', 'PSNR', 'SSIM'],
        )
        df_perRegion.to_csv(os.path.join(out_dir, 'per_region_correlations.csv'), index=False)
        print(f'  Wrote {cell_line} evaluation outputs to {out_dir}')


if __name__ == '__main__':
    main()
