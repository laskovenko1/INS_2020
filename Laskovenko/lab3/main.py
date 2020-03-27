import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def set_plot(figure_num, epoch_num, train_mae, validation_mae):
    x = range(0, epoch_num)
    plt.figure(figure_num)
    plt.plot(x, train_mae, label='Training mean absolute error')
    plt.plot(x, validation_mae, label='Validation mean absolute error')
    plt.title('Absolute error / epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Absolute error')
    plt.legend()
    plt.show()


# Data loading:
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data.shape)
print(test_data.shape)
print(test_targets)

# Data normalization:
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std
test_data -= mean
test_data /= std

# Model building and fitting with K-fold cross-validation:
k = 8
num_val_samples = len(train_data) // k
num_epochs = 40
all_scores = []
val_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0,
                        validation_data=(val_data, val_targets))
    mae_history = history.history["mean_absolute_error"]
    val_mae_history = history.history["val_mean_absolute_error"]
    val_mae_histories.append(val_mae_history)
    set_plot(i, num_epochs, mae_history, val_mae_history)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print(np.mean(all_scores))

average_mae_history = [np.mean([x[i] for x in val_mae_histories]) for i in range(num_epochs)]
plt.figure()
plt.plot(range(0, num_epochs), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel("Mean absolute error")
plt.show()
