import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow_core.python.keras.layers import Dropout

from var2 import gen_data

# Setting constants:
size = 1000
validation_size = size // 10
epochs = 12
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
h = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_data, test_labels))

# Plotting:
loss = h.history['loss']
val_loss = h.history['val_loss']
acc = h.history['acc']
val_acc = h.history['val_acc']

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
