import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer, Dense, Conv1D, Conv2D, Reshape, RepeatVector, Concatenate, Permute, Add
from tensorflow.keras.layers import MultiHeadAttention, BatchNormalization, Embedding, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import repeat_elements
from tensorflow.python.ops import array_ops


class RepeatVector3D(Layer):
    """
    Custom Keras layer that repeats a 3D tensor along a new axis to produce a 4D tensor.

    Input shape:
        (batch_size, x, y)

    Output shape:
        (batch_size, x, n, y)
    """

    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def call(self, inputs):
        # Expand to 4D and repeat along the new axis
        inputs = array_ops.expand_dims(inputs, axis=2)
        return repeat_elements(inputs, self.n, axis=2)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([input_shape[0], input_shape[1], self.n, input_shape[2]])

    def get_config(self):
        config = super().get_config()
        config.update({'n': self.n})
        return config


def CAESAR(nBins=4000, nMarks=14, verbose=1, lr=1e-4,
           positional_dim=8, n_distance=800, distance_dim=8,
           n_SA_layers=2, SA_dim=96, SA_head=8,
           n_Conv_layers=3, Conv_dim=64, Conv_kernel=21,
           n_final_layers=3, final_layer_dim=24, final_layer_kernel=3):
    """
    CAESAR: A custom deep learning model combining Conv1D, self-attention, and pairwise interaction modeling.

    Args:
        nBins: Number of bins or sequence positions
        nMarks: Number of epigenetic marks per bin
        verbose: Whether to print model summary and weights
        lr: Learning rate
        positional_dim: Positional encoding dimension
        n_distance: Max distance for embedding
        distance_dim: Embedding size for distances
        n_SA_layers: Number of self-attention layers
        SA_dim: Attention dimension
        SA_head: Number of attention heads
        n_Conv_layers: Number of 1D conv layers
        Conv_dim: Filter size of Conv1D layers
        Conv_kernel: Kernel size of Conv1D layers
        n_final_layers: Number of final 2D Conv layers
        final_layer_dim: Filter size of 2D Conv layers
        final_layer_kernel: Kernel size of 2D Conv layers

    Returns:
        A compiled Keras model
    """
    assert SA_dim % SA_head == 0, "SA_dim must be divisible by SA_head"
    assert n_SA_layers > 0 and n_Conv_layers > 0 and n_final_layers > 0

    # Inputs
    epi_inp = Input(shape=(nBins, nMarks), name='Inp_epi')
    positional = Input(shape=(nBins, positional_dim), name='Inp_pos')
    distance0 = Input(shape=(nBins, nBins), name='Inp_dis')
    weights = Input(shape=(nBins, nBins), name='Inp_w')

    # Positional + epigenetic concatenation
    epi_data = Concatenate(axis=-1, name='Concat_epi_pos')([epi_inp, positional])

    # 1D Convolutional layers
    x = BatchNormalization(name='Conv_0_BN')(Dense(Conv_dim, activation='relu', name='Conv_0')(epi_data))
    for i in range(n_Conv_layers):
        x = BatchNormalization(name=f'Conv_{i+1}_BN')(
            Conv1D(Conv_dim, Conv_kernel, padding='same', activation='relu', name=f'Conv_{i+1}')(x)
        )

    # Multi-head self-attention layers
    for i in range(n_SA_layers):
        x = BatchNormalization(name=f'SA_{i+1}_BN')(
            MultiHeadAttention(num_heads=SA_head, key_dim=SA_dim // SA_head, name=f'SA_{i+1}')(x, x)
        )

    # Pairwise interaction
    p1 = Conv1D(final_layer_dim, 1, padding='same', activation='relu', name='Pair1')(x)
    p2 = Conv1D(final_layer_dim, 1, padding='same', activation='relu', name='Pair2')(x)
    p1_stack = RepeatVector3D(nBins, name='Repeat_P1')(p1)
    p2_stack = Permute([2, 1, 3], name='Permute_P2')(
        RepeatVector3D(nBins, name='Repeat_P2')(p2)
    )

    # Distance embedding
    distance_flat = Reshape([nBins * nBins], name='Reshape_dis1')(distance0)
    distance_emb = Embedding(n_distance + 1, distance_dim, name='Distance_emb')(distance_flat)
    distance = Reshape([nBins, nBins, distance_dim], name='Reshape_dis2')(distance_emb)

    # Combine pairwise embeddings and distances
    added = Add(name='Add_Pairs')([p1_stack, p2_stack])
    x2 = Concatenate(axis=-1, name='Concat_Pairs_Dist')([added, distance])

    # Final 2D convolutional layers
    for i in range(n_final_layers - 1):
        x2 = BatchNormalization(name=f'FinalConv_{i}_BN')(
            Conv2D(final_layer_dim, final_layer_kernel, padding='same', activation='relu', name=f'FinalConv_{i}')(x2)
        )
    x2 = Conv2D(1, 1, padding='same', activation='relu', name=f'FinalConv_{n_final_layers - 1}')(x2)

    # Final reshape and element-wise multiplication with weights
    outputs = Reshape([nBins, nBins], name='OutputReshape')(x2)
    outputs = Multiply(name='ApplyWeights')([outputs, weights])

    # Build model
    model = Model(inputs=[epi_inp, positional, distance0, weights], outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    if verbose:
        model.summary()
        for name, weight in zip([w.name for l in model.layers for w in l.weights], model.get_weights()):
            print(name, weight.shape)

    return model


if __name__ == '__main__':
    # Example instantiation
    CAESAR(nBins=1000, verbose=1)
