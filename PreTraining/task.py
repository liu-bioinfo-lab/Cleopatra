from os import path
import time
from data_utils import training_data_generator
from CAESAR_model import CAESAR


# Exclude the beginning and end regions which are noisy
HUMAN_CHR_SIZES = {'chr1': (100000, 248900000), 'chr2': (100000, 242100000), 'chr3': (100000, 198200000),
                   'chr4': (100000, 190100000), 'chr5': (100000, 181400000), 'chr6': (100000, 170700000),
                   'chr7': (100000, 159300000), 'chr8': (100000, 145000000), 'chr9': (100000, 138200000),
                   'chr10': (100000, 133700000), 'chr11': (100000, 135000000), 'chr12': (100000, 133200000),
                   'chr13': (100000, 114300000), 'chr14': (100000, 106800000), 'chr15': (100000, 101900000),
                   'chr16': (100000, 90200000), 'chr17': (100000, 83200000), 'chr18': (100000, 80200000),
                   'chr19': (100000, 58600000), 'chr20': (100000, 64400000), 'chr21': (100000, 46700000),
                   'chr22': (100000, 50800000), 'chrX': (100000, 156000000)
                   }


# Paths to contact map folders for different cell lines
FOLDERS = {
    'H1': '../ExampleData/MicroC/H1',
    'HCT-116': '../ExampleData/MicroC/HCT-116',
    'GM12878': '../ExampleData/MicroC/GM12878',
    'K562': '../ExampleData/MicroC/K562'
}

# Path to epigenomic feature files
PATH_epigenomics = '../ExampleData/Epi/'

# Path to OE vectors
PATH_OE = '../ExampleData/MicroC/OE_vectors'


def train_and_evaluate(
        cell_type, epoches=60, batch_size=20, checkpoint_frequency=1,
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
        nMarks=len(epi_names), lr=0.001,
        n_distance=max_range // resolution, nBins=max_range // resolution
    )

    my_path = path.abspath(path.dirname(__file__))

    generator = training_data_generator(
        HUMAN_CHR_SIZES, [cell_type], epi_names,
        max_range=max_range, resolution=resolution, gap=gap,
        n_epoches=epoches, batch_size=batch_size
    )

    for epoch, batch, (epis, pos_enc, distance, weights), micros in generator:
        if batch == 1:
            if epoch != 1 and (epoch - 1) % checkpoint_frequency == 0:
                model.save_weights('{0}/temp_model_{1}.h5'.format(my_path, epoch-1))
        # print(epoch, batch, hics.shape, epis.shape, pos_enc.shape, micros.shape)
        t1 = time.time()
        model.train_on_batch([epis, pos_enc, distance, weights], micros)
        t2 = time.time()
        print(' - Training:', t2 - t1, 's')
        mse = model.evaluate([epis, pos_enc, distance, weights], micros, batch_size=batch_size, verbose=0)
        t3 = time.time()
        print(' - Evaluating:', t3 - t2, 's')
        print(' - MSE:', mse)
    model.save_weights('{0}/temp_model_{1}.h5'.format(my_path, epoches))


if __name__ == '__main__':
    # For 500bp resolution
    train_and_evaluate(
        cell_type='HCT-116',
        epoches=60, batch_size=10, checkpoint_frequency=10,
        max_range=625000, resolution=500, gap=500000
    )

    # For 2000bp resolution
    train_and_evaluate(
        cell_type='HCT-116',
        epoches=60, batch_size=10, checkpoint_frequency=10,
        max_range=2500000, resolution=2000, gap=500000
    )


