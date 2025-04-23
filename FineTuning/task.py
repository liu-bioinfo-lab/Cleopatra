from os import path
import time
from data_utils import training_data_generator
from CAESAR_model import CAESAR


# Path to contact map files
FOLDERS = {
    'H1': '../ExampleData/RCMC/H1',
    'HCT-116': '../ExampleData/RCMC/HCT-116',
    'GM12878': '../ExampleData/RCMC/GM12878',
    'K562': '../ExampleData/RCMC/K562'
}

# Path to loop files (to add weights on loops during training)
LOOPS = {
    'H1': '../ExampleData/Structures/Loops/H1.txt',
    'HCT-116': '../ExampleData/Structures/Loops/HCT-116.txt',
    'GM12878': '../ExampleData/Structures/Loops/GM12878.txt',
    'K562': '../ExampleData/Structures/Loops/K562.txt'
}

# Path to epigenomic feature files
PATH_epigenomics = '../ExampleData/Epi/'

# Path to pretrained model
PATH_pretrained_model = './temp_model_60.h5'

# Path to OE vectors
PATH_OE = '../ExampleData/MicroC/OE_vectors'

# Regions for training
# key: (chromosome, start_loc, end_loc)
TRAINING_REGIONS = {
    0: ('chr6', 25150000, 28600000),  # high
    # 1: ('chr19', 36050000, 39750000),  # exclude
    2: ('chr5', 157000000, 160150000),  # median
    # 3: ('chr1', 207650000, 210350000),  # median
    4: ('chr4', 61350000, 64450000),  # low
    # 5: ('chr8', 125000000, 129750000),  # median
    6: ('chr6', 29750000, 32250000),  # high
    # 7: ('chrX', 47125000, 49375000),  # median high
    8: ('chr1', 237300000, 240500000),  # low
    9: ('chr7', 10500000, 13500000),  # median low
    # 10: ('chr8', 63000000, 66500000),  # median low
    11: ('chr4', 181300000, 184000000),  # median
    12: ('chr3', 118750000, 121500000),  # median
    # 13: ('chr9', 106625000, 109875000)  # median low
}


def train_and_evaluate(
        cell_type, epoches=100, batch_size=20, checkpoint_frequency=1,
        max_range=625000, resolution=500, gap=500000
):
    """
        Train and evaluate the CAESAR model on epigenomic datasets.

        Parameters:
        -----------
        epoches : int
            Number of training epochs (default is 60).
        batch_size : int
            Batch size used during training (default is 20).
        checkpoint_frequency : int
            Frequency (in epochs) to save model checkpoints (default is 1).
        max_range : int
            The max prediction range
        resolution : int
            Resolution
        gap : int
            Sampling gap for training data

        This function initializes the CAESAR model, generates training data
        using `training_data_generator`, trains the model on each batch,
        evaluates the MSE on the same batch, and periodically saves the model
        weights.
    """

    epi_names = ['DNase-seq', 'H3K4me1', 'H3K4me2', 'H3K4me3',
                 'H3K9ac', 'H3K9me3', 'H3K27ac', 'H3K27me3', 'H3K36me3',
                 'H3K79me2', 'H4K20me1', 'CTCF',
                 'EZH2', 'POLR2A', 'JUND', 'REST',
                 'RAD21', 'H2AFZ', 'phastCons']

    # Build current model
    model = CAESAR(
        nMarks=len(epi_names), lr=0.001 / 20,
        n_distance=max_range // resolution, nBins=max_range // resolution
    )
    model.load_weights(PATH_pretrained_model)

    my_path = path.abspath(path.dirname(__file__))

    generator = training_data_generator(
        TRAINING_REGIONS, [cell_type], epi_names,
        max_range=max_range, resolution=resolution, n_distance=n_distance, gap=gap,
        pos_enc_dim=8, n_epoches=epoches, batch_size=batch_size, loop=True
    )

    for epoch, batch, (epis, pos_enc, distance, weights), micros in generator:
        if batch == 1:
            if epoch != 1 and (epoch - 1) % checkpoint_frequency == 0:
                model.save_weights('{0}/new_model_{1}.h5'.format(my_path, epoch-1))
        # print(epoch, batch, hics.shape, epis.shape, pos_enc.shape, micros.shape)
        t1 = time.time()
        model.train_on_batch([epis, pos_enc, distance, weights], micros)
        t2 = time.time()
        print(' - Training:', t2 - t1, 's')
        mse = model.evaluate([epis, pos_enc, distance, weights], micros, batch_size=batch_size, verbose=0)
        t3 = time.time()
        print(' - Evaluating:', t3 - t2, 's')
        print(' - MSE:', mse)
    model.save_weights('{0}/new_model_{1}.h5'.format(my_path, epoches))


if __name__ == '__main__':
    train_and_evaluate(
        'HCT-116', epoches=100, batch_size=10, checkpoint_frequency=20
    )


