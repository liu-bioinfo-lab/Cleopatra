import numpy as np
import tensorflow
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Input, Multiply, Conv1D, Conv2D, Reshape,\
    RepeatVector, Concatenate, Permute, Add, MultiHeadAttention, Activation, BatchNormalization,\
    Embedding, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops import array_ops
from tensorflow.keras.backend import repeat_elements


class RepeatVector3D(Layer):
    """Repeats the input n times.
    Example:
    ```python
    inp = tf.keras.Input(shape=(4,4))
    # now: model.output_shape == (None, 4,4)
    # note: `None` is the batch dimension
    output = RepeatVector3D(3)(inp)
    # now: model.output_shape == (None, 3, 4, 4)
    model = tf.keras.Model(inputs=inp, outputs=output)
    ```
    Args:
      n: Integer, repetition factor.
    Input shape:
      3D tensor of shape `(None, x, y)`.
    Output shape:
      4D tensor of shape `(None, x, n, y)`.
    """

    def __init__(self, n, **kwargs):
        super(RepeatVector3D, self).__init__(**kwargs)
        self.n = n

    def compute_output_shape(self, input_shape):
        input_shape = tensorflow.TensorShape(input_shape).as_list()
        # print(input_shape, tensorflow.TensorShape([input_shape[0], self.n, input_shape[1]]))
        return tensorflow.TensorShape([input_shape[0], self.n, input_shape[1]])

    def call(self, inputs):
        inputs = array_ops.expand_dims(inputs, axis=2)
        # print(inputs.shape)
        repeat = repeat_elements(inputs, self.n, axis=2)
        # print(repeat.shape)
        return repeat

    def get_config(self):
        config = {'n': self.n}
        base_config = super(RepeatVector3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def insulation_kernel(size=41):
    assert size % 2 == 1
    mat = np.ones((size, size))
    half = size // 2
    mat[:half, half + 1:] = -1
    mat[half + 1:, :half] = -1
    mat = tensorflow.constant(mat * 1e-3, dtype=tensorflow.float32)
    mat = tensorflow.reshape(mat, (size, size, 1, 1))
    return mat


ins_kernel = insulation_kernel()


def custom_loss(y_true, y_pred):
    # Define your custom loss function here
    # For example, let's say your custom loss is the absolute difference between y_true and y_pred
    y_true_reshape = tensorflow.reshape(y_true, (-1, 1250, 1250, 1))
    y_pred_reshape = tensorflow.reshape(y_pred, (-1, 1250, 1250, 1))

    conv_true = tensorflow.nn.conv2d(y_true_reshape, ins_kernel, strides=1, padding="VALID")
    conv_pred = tensorflow.nn.conv2d(y_pred_reshape, ins_kernel, strides=1, padding="VALID")

    conv_true_reshape = tensorflow.reshape(conv_true, (-1, 1210, 1210))
    conv_pred_reshape = tensorflow.reshape(conv_pred, (-1, 1210, 1210))

    diag_true = tensorflow.linalg.diag_part(conv_true_reshape)
    diag_pred = tensorflow.linalg.diag_part(conv_pred_reshape)

    loss = tensorflow.reduce_mean(tensorflow.square(diag_true - diag_pred), axis=1)
    return loss


# Combine MSE and custom loss with weights alpha and beta
def combined_loss(y_true, y_pred):
    mse_loss = tensorflow.reduce_mean(tensorflow.square(y_true - y_pred), axis=[1, 2])
    beta = 10.0
    ins_loss = custom_loss(y_true, y_pred)
    # print(mse_loss.shape, ins_loss.shape)
    return mse_loss + beta * ins_loss


def CAESAR(nBins=4000, nMarks=14, verbose=1, lr=0.0001, positional_dim=8,
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
    m.compile(optimizer=Adam(lr=lr), loss=combined_loss)

    if verbose:
        m.summary()
        names = [weight.name for layer in m.layers for weight in layer.weights]
        weights = m.get_weights()
        for name, weight in zip(names, weights):
            print(name, weight.shape)
    return m


if __name__ == '__main__':
    CAESAR(nBins=1000, verbose=1)



