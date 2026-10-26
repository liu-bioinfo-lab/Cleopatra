"""
Cleopatra unified training entry point.

Usage::

    # Pre-train + fine-tune from a single config
    python train.py --config configs/gm12878_500bp.yaml --stage both

    # Only pre-train (Micro-C)
    python train.py --config configs/gm12878_500bp.yaml --stage pretrain

    # Only fine-tune (RCMC), reusing an existing pre-train checkpoint
    python train.py --config configs/gm12878_500bp.yaml --stage finetune \\
                    --pretrain-checkpoint checkpoints/gm12878_500bp_pretrain_60.pt

If ``--stage both`` and the pre-train checkpoint from the config already
exists on disk, the pre-training loop is skipped and fine-tuning starts
from the existing checkpoint.  This lets you rerun fine-tuning quickly
without re-training the (expensive) pre-train stage.
"""
from __future__ import annotations

import argparse
import os
import random
from time import time

import numpy as np
import torch
import yaml


def set_seed(seed):
    """Seed Python / NumPy / Torch (CPU+CUDA) and force deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from cleopatra.model import Cleopatra
from cleopatra.loader import training_data_generator, evaluation_data_generator
from cleopatra.visualize import visualize_HiC_epigenetics
from cleopatra.regions import HUMAN_CHR_SIZES, RCMC_REGIONS


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _paths_for_stage(cfg, stage):
    """Build a data-paths dict for the given stage. Selects PT or FT
    contact/OE roots and keeps shared entries (epigenomics, loops)."""
    p = cfg['paths']
    if stage == 'pretrain':
        contact_root, oe_root = p['pretrain_contact_root'], p['pretrain_oe_root']
    elif stage == 'finetune':
        contact_root, oe_root = p['finetune_contact_root'], p['finetune_oe_root']
    else:
        raise ValueError(stage)
    return {
        'epi_root':      p['epi_root'],
        'contact_root':  contact_root,
        'oe_root':       oe_root,
        'loops_dir':     p.get('loops_dir'),
    }


def _batch_to_tensors(batch_tuple):
    epis, pos_enc, distance, weights = batch_tuple
    return (
        torch.from_numpy(epis.astype(np.float32)),
        torch.from_numpy(pos_enc.astype(np.float32)),
        torch.from_numpy(distance.astype(np.int64)),
        torch.from_numpy(weights.astype(np.float32)),
    )


def _max_range(resolution):
    if resolution == 500:
        return 625000
    if resolution == 2000:
        return 2500000
    raise ValueError(f'resolution must be 500 or 2000 (got {resolution})')


def _build_model(cfg, stage, load_from=None):
    """Instantiate a Cleopatra model with the config's hyperparameters."""
    stage_cfg = cfg[stage]
    arch = cfg['model']
    lr = stage_cfg['learning_rate']
    model = Cleopatra(
        nBins=arch['n_bins'], nMarks=len(cfg['data']['epigenetic_inputs']),
        verbose=1, lr=lr,
        positional_dim=arch['positional_dim'],
        n_distance=arch['n_bins'], distance_dim=arch['distance_dim'],
        n_SA_layers=arch['n_SA_layers'], SA_dim=arch['SA_dim'], SA_head=arch['SA_head'],
        n_Conv_layers=arch['n_Conv_layers'], Conv_dim=arch['Conv_dim'], Conv_kernel=arch['Conv_kernel'],
        n_final_layers=arch['n_final_layers'], final_layer_dim=arch['final_layer_dim'],
        final_layer_kernel=arch['final_layer_kernel'],
        insulation_loss=stage_cfg.get('insulation_loss', False),
        beta=stage_cfg.get('insulation_beta', 1.0),
        freeze_distance_embedding=stage_cfg.get('freeze_distance_embedding', False),
        rbf_n_centers=arch['rbf_n_centers'], rbf_width_factor=arch['rbf_width_factor'],
    )
    if load_from is not None:
        print(f'Loading weights from {load_from}')
        model.load_weights(load_from, device=DEVICE)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def _train_loop(model, optimizer, generator, checkpoint_prefix, checkpoint_frequency, n_epochs):
    for epoch, batch_idx, batch_tuple, targets, keys in generator:
        if batch_idx == 0 and epoch != 0 and epoch % checkpoint_frequency == 0:
            path = f'{checkpoint_prefix}_{epoch}.pt'
            model.save_weights(path)
            print(f'  Saved checkpoint: {path}')
        t1 = time()
        batch_tensors = _batch_to_tensors(batch_tuple)
        targets_t = torch.from_numpy(targets.astype(np.float32)).to(DEVICE)
        model.train_step(batch_tensors, targets_t, optimizer, DEVICE)
        mse = model.evaluate_batch(batch_tensors, targets_t, DEVICE)
        print(f'  batch time: {time() - t1:.1f}s  MSE: {mse:.4f}')
    final = f'{checkpoint_prefix}_{n_epochs}.pt'
    model.save_weights(final)
    print(f'  Saved final checkpoint: {final}')
    return final


def _visualize(model, cfg, stage_cfg, stage, mode, cell_lines, out_dir, resolution, max_range, epi_names):
    os.makedirs(out_dir, exist_ok=True)
    generator = evaluation_data_generator(
        RCMC_REGIONS, cell_lines, epi_names, {}, mode=mode,
        paths=_paths_for_stage(cfg, stage),
        max_range=max_range, resolution=resolution,
        n_distance=cfg['model']['n_bins'],
        gap=stage_cfg['gap'],
        pos_enc_dim=cfg['model']['positional_dim'],
        n_epoches=1, batch_size=1,
        smooth=stage_cfg.get('smooth_inputs', True),
    )
    for _, _, batch_tuple, targets, keys, _ in generator:
        pred = model.predict_batch(_batch_to_tensors(batch_tuple), DEVICE)
        _res = pred[0]; _epi = batch_tuple[0][0].T; _ww = batch_tuple[3][0]
        _tgt = targets[0]; ky = keys[0]
        visualize_HiC_epigenetics(
            [_tgt, _res, _ww], [_epi, _epi, _epi],
            os.path.join(out_dir, f'vis_{ky}.png'),
            colorbar=True, interval_after_heatmap=0., interval_between_epi=0.,
            epi_labels=epi_names, epi_yaxis=True, fontsize=20, epi_ratio=0.045,
            x_ticks=[''] * 6, vmax=np.quantile(_tgt, 0.99),
        )


def pretrain(cfg, args):
    resolution = cfg['data']['resolution']
    max_range = _max_range(resolution)
    stage_cfg = cfg['pretrain']

    print(f'=== Pre-training ({resolution} bp, {stage_cfg["n_epochs"]} epochs) ===')
    model, optimizer = _build_model(cfg, 'pretrain')

    skip = RCMC_REGIONS if stage_cfg.get('exclude_rcmc_regions', True) else {}
    generator = training_data_generator(
        HUMAN_CHR_SIZES, cfg['data']['cell_lines'], cfg['data']['epigenetic_inputs'],
        skip, mode='PT', paths=_paths_for_stage(cfg, 'pretrain'),
        max_range=max_range, resolution=resolution,
        n_distance=cfg['model']['n_bins'],
        gap=stage_cfg['gap'],
        pos_enc_dim=cfg['model']['positional_dim'],
        n_epoches=stage_cfg['n_epochs'], batch_size=cfg['batch_size'],
        distance_weight=stage_cfg.get('distance_weight', True),
        loop=stage_cfg.get('loop_weight', False),
        distance_weight_rule=stage_cfg.get('distance_weight_rule'),
        smooth=stage_cfg.get('smooth_inputs', True),
    )
    checkpoint = _train_loop(
        model, optimizer, generator,
        args.pretrain_prefix, cfg['checkpoint_frequency'], stage_cfg['n_epochs'],
    )
    if stage_cfg.get('visualize', False):
        _visualize(
            model, cfg, stage_cfg, stage='pretrain', mode='PT',
            cell_lines=cfg['data']['cell_lines'],
            out_dir='pretrain_figures', resolution=resolution,
            max_range=max_range, epi_names=cfg['data']['epigenetic_inputs'],
        )
    return checkpoint


def finetune(cfg, args, pretrain_checkpoint):
    resolution = cfg['data']['resolution']
    max_range = _max_range(resolution)
    stage_cfg = cfg['finetune']

    print(f'=== Fine-tuning ({resolution} bp, {stage_cfg["n_epochs"]} epochs) ===')
    model, optimizer = _build_model(cfg, 'finetune', load_from=pretrain_checkpoint)

    regions_to_use = stage_cfg['training_regions']
    ft_regions = {k: RCMC_REGIONS[k] for k in regions_to_use}
    generator = training_data_generator(
        ft_regions, cfg['data']['cell_lines'], cfg['data']['epigenetic_inputs'],
        {}, mode='FT', paths=_paths_for_stage(cfg, 'finetune'),
        max_range=max_range, resolution=resolution,
        n_distance=cfg['model']['n_bins'],
        gap=stage_cfg['gap'],
        pos_enc_dim=cfg['model']['positional_dim'],
        n_epoches=stage_cfg['n_epochs'], batch_size=cfg['batch_size'],
        distance_weight=stage_cfg.get('distance_weight', True),
        loop=stage_cfg.get('loop_weight', True),
        loop_mode=stage_cfg.get('loop_mode', 'compound'),
        distance_weight_rule=stage_cfg.get('distance_weight_rule'),
        smooth=stage_cfg.get('smooth_inputs', True),
    )
    checkpoint = _train_loop(
        model, optimizer, generator,
        args.finetune_prefix, cfg['checkpoint_frequency'], stage_cfg['n_epochs'],
    )
    if stage_cfg.get('visualize', True):
        _visualize(
            model, cfg, stage_cfg, stage='finetune', mode='FT',
            cell_lines=cfg['data']['cell_lines'],
            out_dir='finetune_figures', resolution=resolution,
            max_range=max_range, epi_names=cfg['data']['epigenetic_inputs'],
        )
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--stage', choices=['pretrain', 'finetune', 'both'], default='both')
    parser.add_argument('--pretrain-checkpoint', default=None,
                        help='Existing PT checkpoint. Only used with --stage finetune. '
                             'If omitted, --pretrain-prefix + n_epochs is used.')
    parser.add_argument('--pretrain-prefix', default='pretrain',
                        help='Filename prefix for PT checkpoints (default: pretrain -> pretrain_60.pt)')
    parser.add_argument('--finetune-prefix', default='finetune',
                        help='Filename prefix for FT checkpoints (default: finetune -> finetune_200.pt)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for Python/NumPy/Torch. Omit for stochastic runs.')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print(f'Loaded config: {args.config}')
    print(f'Device: {DEVICE}')
    if args.seed is not None:
        set_seed(args.seed)
        print(f'Seeded RNGs with {args.seed} (deterministic cuDNN enabled)')

    pt_ckpt = None
    if args.stage in ('pretrain', 'both'):
        expected_pt = f'{args.pretrain_prefix}_{cfg["pretrain"]["n_epochs"]}.pt'
        if args.stage == 'both' and os.path.exists(expected_pt):
            print(f'Skipping pre-training: {expected_pt} already exists.')
            pt_ckpt = expected_pt
        else:
            pt_ckpt = pretrain(cfg, args)

    if args.stage in ('finetune', 'both'):
        if args.pretrain_checkpoint is not None:
            pt_ckpt = args.pretrain_checkpoint
        elif pt_ckpt is None:
            pt_ckpt = f'{args.pretrain_prefix}_{cfg["pretrain"]["n_epochs"]}.pt'
        if not os.path.exists(pt_ckpt):
            raise FileNotFoundError(
                f'Pre-train checkpoint not found: {pt_ckpt}. '
                'Run with --stage pretrain or pass --pretrain-checkpoint.'
            )
        finetune(cfg, args, pt_ckpt)


if __name__ == '__main__':
    main()
