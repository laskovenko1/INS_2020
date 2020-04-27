import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, GRU, LSTM
from keras.optimizers import Adam

from data import gen_data_from_sequence


def plot_results(predicted_res, test_res):
    pred_length = range(len(predicted_res))
    plt.title('Sequence')
    plt.ylabel('Sequence')
    plt.xlabel('x')
    plt.plot(pred_length, predicted_res)
    plt.plot(pred_length, test_res)
    plt.show()


data, res = gen_data_from_sequence()
train_size = (len(data) // 10) * 8
val_size = (len(data) - train_size) // 2

train_data, train_res = data[:train_size], res[:train_size]
val_data, val_res = data[train_size:train_size + val_size], res[train_size:train_size + val_size]
test_data, test_res = data[train_size + val_size:], res[train_size + val_size:]

model = Sequential()
model.add(GRU(32, recurrent_activation='sigmoid', input_shape=(None, 1), return_sequences=True))
model.add(LSTM(64, activation='relu', input_shape=(None, 1), return_sequences=True, dropout=0.5))
model.add(GRU(32, input_shape=(None, 1), recurrent_dropout=0.2))
model.add(Dense(1))
model.compile(Adam(), loss='mse')
H = model.fit(train_data,
              train_res,
              batch_size=9,
              epochs=20,
              verbose=1,
              validation_data=(val_data, val_res))

predict_res = model.predict(test_data)
plot_results(predict_res, test_res)
