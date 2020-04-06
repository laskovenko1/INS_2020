import numpy as np
import pandas as pd
from keras import Input, Model
from keras.layers import Dense

TRAIN_DATA_SIZE = 500
TEST_DATA_SIZE = 50


def generate_data(N):
    data = []
    targets = []
    for i in range(N):
        X = np.random.normal(3, 10)
        e = np.random.normal(0, 0.3)
        data.append([X ** 2 + e,
                     np.cos(2 * X) + e,
                     X - 3 + e,
                     -X + e,
                     np.abs(X) + e,
                     (X ** 3) / 4 + e])
        targets.append([np.sin(X / 2) + e])
    return np.array(data), np.array(targets)


def normalize_data(data):
    mean = data.mean(axis=0)
    data -= mean
    std = data.std(axis=0)
    data /= std


# Generating random data:
train_data, train_targets = generate_data(TRAIN_DATA_SIZE)
test_data, test_targets = generate_data(TEST_DATA_SIZE)

# Exporting generated data to .csv:
pd.DataFrame(np.round(train_data, decimals=2)).to_csv("train_data.csv")
pd.DataFrame(np.round(train_targets, decimals=2)).to_csv("train_targets.csv")
pd.DataFrame(np.round(test_data, decimals=2)).to_csv("test_data.csv")
pd.DataFrame(np.round(test_targets, decimals=2)).to_csv("test_targets.csv")

# Normalizing data:
normalize_data(train_data)
normalize_data(test_data)

# Building model:
main_input = Input(shape=(6,))

encoded = Dense(32, activation='relu')(main_input)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(3, activation='relu')(encoded)

decoded_input = Input(shape=(3,))
decoded = Dense(64, activation='relu', name='decoded_layer_1')(encoded)
decoded = Dense(64, activation='relu', name='decoded_layer_2')(decoded)
decoded = Dense(6, name='decoded_output')(decoded)

predicted = Dense(32, activation='relu')(encoded)
predicted = Dense(64, activation='relu')(predicted)
predicted = Dense(1)(predicted)

# Encoding model:
encoder = Model(inputs=main_input, outputs=encoded)
encoder.save('encoder.h5')
# Decoding model:
#   Fitting autoencoder:
autoencoder = Model(inputs=main_input, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
autoencoder.fit(train_data, train_data, epochs=100, batch_size=10, validation_data=(test_data, test_data))
#   Building decoder:
decoder = autoencoder.get_layer('decoded_layer_1')(decoded_input)
decoder = autoencoder.get_layer('decoded_layer_2')(decoder)
decoder = autoencoder.get_layer('decoded_output')(decoder)
decoder = Model(decoded_input, decoder)
decoder.save('decoder.h5')
# Predicting model:
predictor = Model(inputs=main_input, outputs=predicted)
#   Fitting predicting model:
predictor.compile(optimizer='adam', loss='mse', metrics=['mae'])
predictor.fit(train_data, train_targets, epochs=100, batch_size=10, validation_data=(test_data, test_targets))
predictor.save('predictor.h5')

# Encoding data:
encoded_data = encoder.predict(test_data)
pd.DataFrame(np.round(encoded_data, decimals=2)).to_csv("encoded_data.csv")

# Decoding data:
decoded_data = decoder.predict(encoded_data)
pd.DataFrame(np.round(decoded_data, decimals=2)).to_csv("decoded_data.csv")

# Predicted data:
predicted_data = predictor.predict(test_data)
pd.DataFrame(np.round(predicted_data, decimals=2)).to_csv("predicted_data.csv")
