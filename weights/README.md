# Pre-trained weights

## `gm12878_500bp_finetune.pt`

Canonical GM12878 model at 500 bp resolution, produced by the config
`configs/gm12878_500bp.yaml` after 60 pre-training epochs (Micro-C) plus
200 fine-tuning epochs (RCMC). All hyperparameters match the published
manuscript.

Load with:

```python
import yaml, torch
from cleopatra.model import Cleopatra

with open('configs/gm12878_500bp.yaml') as f:
    cfg = yaml.safe_load(f)
arch = cfg['model']

model = Cleopatra(
    nBins=arch['n_bins'], nMarks=len(cfg['data']['epigenetic_inputs']),
    positional_dim=arch['positional_dim'],
    n_distance=arch['n_bins'], distance_dim=arch['distance_dim'],
    n_SA_layers=arch['n_SA_layers'], SA_dim=arch['SA_dim'], SA_head=arch['SA_head'],
    n_Conv_layers=arch['n_Conv_layers'], Conv_dim=arch['Conv_dim'], Conv_kernel=arch['Conv_kernel'],
    n_final_layers=arch['n_final_layers'], final_layer_dim=arch['final_layer_dim'],
    final_layer_kernel=arch['final_layer_kernel'],
    rbf_n_centers=arch['rbf_n_centers'], rbf_width_factor=arch['rbf_width_factor'],
    insulation_loss=False, freeze_distance_embedding=True,
)
model.load_weights('weights/gm12878_500bp_finetune.pt')
model.eval()
```

## Other checkpoints (H1, HCT-116, K562, 2 kb variants)

Additional cell-type / resolution checkpoints and genome-wide predicted
contact maps are available on Zenodo:

- Predictions (all cell types, both resolutions): **10.5281/zenodo.15299993**
- RCMC / Micro-C source data: **10.5281/zenodo.15303879**

To reproduce a checkpoint from scratch, edit a copy of
`configs/gm12878_500bp.yaml` (change `data.cell_lines`, adjust
`paths.*_root`) and run:

```bash
python train.py --config configs/<your_config>.yaml --stage both
```
