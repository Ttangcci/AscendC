import numpy as np

features = np.array([
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [5.0, 4.0, 3.0, 2.0, 1.0]
])
labels = np.array([
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0]
], dtype=float)

softmax = np.exp(features - features.max(axis=1, keepdims=True))
softmax /= softmax.sum(axis=1, keepdims=True)
loss = -np.sum(labels * np.log(softmax), axis=1)
backprop = softmax - labels

print("loss:", loss)
print("backprop:", backprop)