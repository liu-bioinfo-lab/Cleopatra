# Cleopatra

---

**Cleopatra** is a deep learning framework designed for the unique characteristics of ultra-deep **RCMC (Region-Capture Micro-C)** data. By leveraging a comprehensive panel of one-dimensional (1D) genomic features, Cleopatra predicts high-resolution 3D genome architecture.

Traditional models for predicting chromatin contact maps require large training datasets across many chromosomes, which are often impractical to generate with experimental RCMC data. To overcome this limitation, Cleopatra uses a two-stage training strategy: it is first pre-trained on whole-genome Micro-C data, and then fine-tuned on ultra-deep RCMC data. This strategy enables Cleopatra to transfer knowledge from abundant but lower-resolution data to accurately impute high-resolution genome-wide contact maps.

---

## Requirements

Ensure the following Python environment is set up before running Cleopatra:

- Python 3.8.8  
- TensorFlow 2.4.1  
- NumPy 1.19.2  
- pandas 1.2.3  
- SciPy 1.6.1  

---

## Data Preparation

Cleopatra accepts **19 one-dimensional (1D) genomic features** as input, including chromatin accessibility, histone modifications, and transcription factor ChIP-seq data.
The output is Micro-C contact map (pre-training) or RCMC contact map (fine-tuning).

The folders for training data should follow the structure below.
**Examples are also under the `/ExampleData` folder.**

```
ExampleData
├── MicroC
│   ├── HCT-116
│   │   ├── chr1_2000bp.txt
│   │   ├── chr1_500bp.txt
│   │   ├── ...
│   │   └── chrX_500bp.txt
│   ├── GM12878
│   │   ├── chr1_2000bp.txt
│   │   └── ...
│   ├── ...
│   └── OE_vectors
│       ├── HCT-116_500bp.npy
│       ├── ...
│       └── GM12878_2000bp.npy
├── RCMC
│   ├── HCT-116
│   │   ├── chr1_2000bp.txt
│   │   ├── ...
│   ├── GM12878
│   │   ├── chr1_2000bp.txt
│   │   └── ...
│   ├── ...
│   └── OE_vectors
│       ├── HCT-116_500bp_0.npy
│       ├── ...
│       └── GM12878_2000bp_13.npy
└── Epigenomics
    ├── HCT-116
    │   ├── chr1
    │   │   ├── chr1_2000bp_H3K4me1.npy
    │   │   ├── chr1_500bp_RAD21.npy
    │   │   └── ...
    │   ├── chr2
    │   │   └── ...
    │   └── chrX
    │       └── ...
    └── GM12878
        ├── chr1
        └── ...
```

- Each contact file (e.g., `chr1_2000bp.txt`) should be generated using the `dump` function of **cooler**:
  ```bash
  cooler dump -r "chr${chr}" -b --join -o "${OUTPUT_FILE}" "${COOLER_FILE}"
  ```

- The OE (observed/expected) vectors store the average contact values for each contact distance, which is used for OE normalization.
  - For Micro-C, each cell type has only one "genome-wide" OE vector at a certain resolution (`{cell_type}_{resolution}bp.npy`)
  - For RCMC, due to diversities of regions, each region has a different OE vector (`{cell_type}_{resolution}bp_{region_index}.npy`)

- Each epigenomic track is a `.npy` file generated using the **pyBigWig** Python package. Refer to the supplementary table of the paper for the download link of the ENCODE bigWig files.

---

## Model Training

Cleopatra uses the same model architecture for both pre-training (Micro-C) and fine-tuning (RCMC). The main components are:

```
PreTraining/FineTuning
├── task.py             # Main training script
├── CAESAR_model.py     # Model architecture
├── config.py           # Configurations (mostly data paths)
└── data_utils.py       # Data loading and preprocessing
```

Before running, update the following parameters in `config.py`:
- `FOLDERS`: Paths to the contact map files  
- `PATH_epigenomics`: Paths to the epigenomic feature files  
- `PATH_OE`: Paths to OE vectors
- `TRAINING_REGION` (fine-tune only): which RCMC regions are used for training
- `LOOPS` (fine-tune only, optional): Paths to loops. If given, the model will give higher weights on pre-annotated loops

The main entry point is `task.py`.
To begin training, run the function `train_and_evaluate()` in `task.py`:

```Python
train_and_evaluate(
    cell_type='HCT-116',
    epoches=60, batch_size=10, checkpoint_frequency=10,
    max_range=625000, resolution=500, gap=500000
)
```
- `cell_type`: Which cell type to train
- `epoch`: Number of training epochs
- `batch_size`: Batch size used during training
- `checkpoint_frequency`: Frequency (in epochs) to save model checkpoints
- `max_range`: Prediction window (e.g., 625kb for 500bp resolution, 2.5Mb for 2kb resolution)  
- `resolution`: Either 500 or 2000 (make sure the input files match this)  
- `gap`: Sampling gap for generating training instances


---

## Citation

If you use Cleopatra in your research, please cite:

```
[Insert preprint here]
```

---

## Contact

For questions, issues, or contributions, feel free to open an issue or email `fanfeng@umich.edu`!
