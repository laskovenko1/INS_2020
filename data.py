import numpy as np
import random
import math
import matplotlib.pyplot as plt


def gen_sequence(seq_len=1000):
    seq = [math.sin(i / 5) / 2 + math.cos(i / 3) / 2 + random.normalvariate(0, 0.04) for i in range(seq_len)]
    return np.array(seq)


def draw_sequence():
    seq = gen_sequence(100)
    plt.plot(range(len(seq)), seq)
    plt.show()


def gen_data_from_sequence(seq_len=1006, lookback=10):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i, i + lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback, len(seq))])
    return past, future
