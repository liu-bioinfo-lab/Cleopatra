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


def Cleopatra(nBins=4000, nMarks=14, verbose=1, lr=0.0001, positional_dim=8,
           n_distance=800, distance_dim=8,
           n_SA_layers=2, SA_dim=96, SA_head=8,
           n_Conv_layers=3, Conv_dim=64, Conv_kernel=21,
           n_final_layers=3, final_layer_dim=24, final_layer_kernel=3,
           ):
    assert SA_dim % SA_head == 0
    assert n_SA_layers > 0
    assert n_Conv_layers > 0
    assert n_final_layers > 0
    # Inputs
    epi_inp = Input(shape=(nBins, nMarks), name='Inp_epi')
    positional = Input(shape=(nBins, positional_dim), name='Inp_pos')
    distance0 = Input(shape=(nBins, nBins), name='Inp_dis')
    weights = Input(shape=(nBins, nBins), name='Inp_w')

    distance = Reshape([nBins * nBins], name='Reshape_dis1')(distance0)
    epi_data = Concatenate(axis=-1, name='conc')([epi_inp, positional])

    # Conv1D layers
    conv_layers = [
        epi_data,
        BatchNormalization(name=f'Conv_0_BN')(
            Dense(Conv_dim, activation='relu', name=f'Conv_0')(epi_data)
        )
    ]
    for i in range(n_Conv_layers):
        conv_layers.append(
            BatchNormalization(name=f'Conv_{i + 1}_BN')(
                Conv1D(filters=Conv_dim, kernel_size=Conv_kernel, padding='same',
                       name=f'Conv_{i + 1}', activation='relu')(conv_layers[-1])
            )
        )

    # self-attention Layers
    for i in range(n_SA_layers):
        conv_layers.append(
            BatchNormalization(name=f'SA_{i + 1}_BN')(
                MultiHeadAttention(num_heads=SA_head, key_dim=SA_dim // SA_head,
                                   name=f'SA_{i + 1}')(conv_layers[-1], conv_layers[-1])
            )
        )

    # Conc layers
    conc_output = Concatenate(axis=-1, name=f'Conc')(conv_layers)

    # Pairwise combination
    p1 = Conv1D(filters=final_layer_dim, kernel_size=1, padding='same',
                activation='relu', name=f'Pair1')(conc_output)
    p2 = Conv1D(filters=final_layer_dim, kernel_size=1, padding='same',
                activation='relu', name=f'Pair2')(conc_output)
    p1_stack = RepeatVector3D(nBins, name=f'Pair1_stack')(p1)
    p2_stack = RepeatVector3D(nBins, name=f'Pair2_stack')(p2)
    p2_stack_T = Permute(dims=[2, 1, 3], name=f'Pair2_stack_T')(p2_stack)

    distance = Embedding(n_distance+1, distance_dim, name='Distance_emb')(distance)
    distance = Reshape([nBins, nBins, distance_dim], name='Reshape_dis2')(distance)

    adds = Add(name='Pairs_combine')([p1_stack, p2_stack_T])  # 1250 * 1250 * 96

    final_layers = [
        Concatenate(axis=-1, name=f'Emb_dis')([adds, distance])
    ]
    for i in range(n_final_layers - 1):
        final_layers.append(
            BatchNormalization(name=f'final_{i}_BN')(
                Conv2D(filters=final_layer_dim, kernel_size=final_layer_kernel, padding='same',
                       activation='relu', name=f'final_{i}')(final_layers[-1])
            )
        )
    outputs = Conv2D(filters=1, kernel_size=1, padding='same',
                     name=f'final_{n_final_layers - 1}', activation='relu')(final_layers[-1])
    outputs = Reshape([nBins, nBins], name='FinalReshape')(outputs)
    outputs = Multiply(name='FinalWeights')([outputs, weights])

    m = Model(inputs=[epi_inp, positional, distance0, weights], outputs=outputs)
    m.compile(optimizer=Adam(lr=lr), loss='mse')

    if verbose:
        m.summary()
        names = [weight.name for layer in m.layers for weight in layer.weights]
        weights = m.get_weights()
        for name, weight in zip(names, weights):
            print(name, weight.shape)
    return m
