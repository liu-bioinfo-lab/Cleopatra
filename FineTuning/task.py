from os import path
import time
from data_utils import training_data_generator
from CAESAR_model import CAESAR
from config import PATH_pretrained_model, TRAINING_REGIONS


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
        max_range=max_range, resolution=resolution, gap=gap,
        pos_enc_dim=8, n_epoches=epoches, batch_size=batch_size
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


