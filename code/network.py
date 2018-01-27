# Stacked LSTM with residual connections in depth direction.
#
# Naturally LSTM has something like residual connections in time.
# Here we add residual connection in depth.
#
# Inspired by Google's Neural Machine Translation System (https://arxiv.org/abs/1609.08144).
# They observed that residual connections allow them to use much deeper stacked RNNs.
# Without residual connections they were limited to around 4 layers of depth.
#
# It uses Keras 2 API.

from keras.layers import LSTM, Lambda
from keras.layers.merge import add

def make_residual_lstm_layers(input, rnn_width, rnn_depth, rnn_dropout):
    """
    The intermediate LSTM layers return sequences, while the last returns a single element.
    The input is also a sequence. In order to match the shape of input and output of the LSTM
    to sum them we can do it only for all layers but the last.
    """
    x = input
    for i in range(rnn_depth):
        return_sequences = i < rnn_depth - 1
        x_rnn = LSTM(rnn_width, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=return_sequences)(x)
        if return_sequences:
            # Intermediate layers return sequences, input is also a sequence.
            if i > 0 or input.shape[-1] == rnn_width:
                x = add([x, x_rnn])
            else:
                # Note that the input size and RNN output has to match, due to the sum operation.
                # If we want different rnn_width, we'd have to perform the sum from layer 2 on.
                x = x_rnn
        else:
            # Last layer does not return sequences, just the last element
            # so we select only the last element of the previous output.
            def slice_last(x):
                return x[..., -1, :]
            x = add([Lambda(slice_last)(x), x_rnn])
    return x

if __name__ == '__main__':
    # Example usage
    from keras.layers import Input, Dropout, Dense
    from keras.models import Model
    
    input = Input(shape=(64, 5))
    output = make_residual_lstm_layers(input, rnn_width=128, rnn_depth=4, rnn_dropout=0.2)
    output = Dropout(0.2)(output)
    outputSoftmax = Dense(3, activation='softmax')(output)
    outputTanh = Dense(1, activation='tanh')(output)
    model = Model(inputs=input, outputs=[outputSoftmax, outputTanh])
    model.summary()
    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mse'], metrics=['acc'])
    model.fit(x_train, y_train, epochs=1, batch_size=128, validation_data=(x_valid, y_valid), verbose=1)
    model.save('./model.h5')

