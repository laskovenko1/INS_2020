import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout

from var2 import gen_data


# Defining custom callbacks:
class AccuracyHistogramsBuilding(tf.keras.callbacks.Callback):
    def __init__(self, on_epochs):
        super(AccuracyHistogramsBuilding, self).__init__()
        self.on_epochs = on_epochs
        self.accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.on_epochs:
            self.accuracy.append((logs.get('accuracy'), logs.get('val_accuracy')))

    def on_train_end(self, logs=None):
        plt.bar(list(map(lambda x: x + 0.85, self.on_epochs)), list(map(lambda x: x[0], self.accuracy)), width=0.5, label='training acc')
        plt.bar(list(map(lambda x: x + 1.15, self.on_epochs)), list(map(lambda x: x[1], self.accuracy)), width=0.5, label='validation acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(prop={'size': 10})
        plt.title('Training and validation accuracy')
        plt.savefig("acc_hist")
        plt.show()


class ModelsSaving(tf.keras.callbacks.Callback):
    def __init__(self, on_epochs):
        super(ModelsSaving, self).__init__()
        self.on_epochs = on_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.on_epochs:
            self.model.save(str(epoch+1) + ".h5")


# Setting constants:
size = 1000
validation_size = size // 10
epochs = 12
on_epochs = [1, 4, 6, 8, 9, 10, 12]
batch_size = 90

# Generating data:
data, labels = gen_data(size)

# Encoding string labels with values in [0, len(labels) - 1]:
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# Shuffling data:
data_with_labels = list(map(lambda x, y: (x, y), data, labels))
np.random.shuffle(data_with_labels)
data = np.asarray(list(map(lambda x: x[0], data_with_labels)))
labels = np.asarray(list(map(lambda x: x[1], data_with_labels)))

# Reshaping data:
data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))

# Splitting data:
train_data = data[validation_size:]
train_labels = labels[validation_size:]
test_data = data[:validation_size]
test_labels = labels[:validation_size]

# Building model:
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(50, 50, 1)))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitting model:
h = model.fit(train_data,
              train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(test_data, test_labels),
              callbacks=[AccuracyHistogramsBuilding(list(map(lambda x: x - 1, on_epochs))),
                         ModelsSaving(list(map(lambda x: x - 1, on_epochs)))])

# Plotting:
loss = h.history['loss']
val_loss = h.history['val_loss']
acc = h.history['accuracy']
val_acc = h.history['val_accuracy']

plt.figure(1)
plt.plot(range(1, epochs + 1), loss, 'b', label='Training loss')
plt.plot(range(1, epochs + 1), val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure(2)
plt.clf()
plt.plot(range(1, epochs + 1), acc, 'b', label='Training acc')
plt.plot(range(1, epochs + 1), val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
