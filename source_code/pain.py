import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, LSTM, Flatten, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import model_to_dot
from IPython.display import Image

def Siamese_Neural_Network(input_shape=(2, 1024), lstm_units=100, cnn_filters=64, dense_units=64):
    input_layer = Input(shape=input_shape, name='Input')

    # Reshape the input to (time_steps, features) for Conv2D
    reshaped_input = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1), name='Reshape_Input')(input_layer)

    # CNN Branch
    cnn_branch = Conv2D(filters=cnn_filters, kernel_size=(3, 1), activation='relu', padding='same', name='CNN_Conv1')(reshaped_input)
    cnn_branch = MaxPooling2D(pool_size=(2, 1), padding='same', name='CNN_MaxPool1')(cnn_branch)
    cnn_branch = Conv2D(filters=128, kernel_size=(3, 1), activation='relu', padding='same', name='CNN_Conv2')(cnn_branch)
    cnn_branch = MaxPooling2D(pool_size=(2, 1), padding='same', name='CNN_MaxPool2')(cnn_branch)
    cnn_branch = Flatten(name='CNN_Flatten')(cnn_branch)

    # RNN Branch 1
    rnn_branch_1 = LSTM(lstm_units, return_sequences=True, name='RNN_LSTM1')(input_layer)
    rnn_branch_1 = LSTM(lstm_units, name='RNN_LSTM2')(rnn_branch_1)

    # RNN Branch 2 (identical to Branch 1)
    rnn_branch_2 = LSTM(lstm_units, return_sequences=True, name='RNN_LSTM3')(input_layer)
    rnn_branch_2 = LSTM(lstm_units, name='RNN_LSTM4')(rnn_branch_2)

    # Concatenate branches
    merged = concatenate([cnn_branch, rnn_branch_1, rnn_branch_2], axis=-1, name='Concatenate')

    # Add Dense layers for deeper representation learning
    merged = Dense(256, activation='relu', name='Dense1')(merged)
    merged = Dense(128, activation='relu', name='Dense2')(merged)

    # Softmax classifier
    output_layer = Dense(2, activation='softmax', name='Output')(merged)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create the Siamese Neural Network model
siamese_model = Siamese_Neural_Network()

# Plot the model architecture using pydot
plot_model(siamese_model, to_file='siamese_model111.png', show_shapes=True, show_layer_names=True)

# Display the model architecture
Image('siamese_model111.png')

