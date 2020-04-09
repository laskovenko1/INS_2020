import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import *


def build_model(optimizer):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_image(path):
    image = Image.open(path)
    image = image.resize((28, 28))
    image = np.dot(np.asarray(image), np.array([1 / 3, 1 / 3, 1 / 3]))
    image /= 255
    image = 1 - image
    image = image.reshape((1, 28, 28))
    return image


# Loading MNIST data:
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizing data:
train_images = train_images / 255.0
test_images = test_images / 255.0

# Labels coding:
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Building model:
model = build_model(Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999))
history = model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_data=(test_images, test_labels))

# Plotting:
plt.title('Training and test loss')
plt.plot(history.history['loss'], 'r', label='train')
plt.plot(history.history['val_loss'], 'b', label='test')
plt.legend()
plt.show()
plt.clf()
plt.title('Training and test accuracy')
plt.plot(history.history['acc'], 'r', label='train')
plt.plot(history.history['val_acc'], 'b', label='test')
plt.legend()
plt.show()
plt.clf()

# Test accuracy:
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

# Loading custom image:
image = load_image('6.png')
res = model.predict(image)
print(np.argmax(res))
