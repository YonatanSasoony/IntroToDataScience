import sys

import numpy as np
import struct
from array import array
from os.path import join
import os
import random
import matplotlib.pyplot as plt
from MNISTReader import MnistDataloader

cwd = os.getcwd()
input_path = cwd + '\MNIST'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte\\train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte\\train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte\\t10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte')

k = 10
d = 28*28  # 784


def initCenters():
    centers = np.empty(k, dtype=object)
    for i in range(k):
        centers[i] = (np.random.rand(d, 1))
    return centers


def isConverged(prevCenters, centers):
    return np.array_equal(prevCenters, centers)


def subtract(a, b):
    result = np.zeros(d)
    for i in range(d):
        result[i] = a[i] - b[i]
    return result


def calcClusters(X, centers):
    clusters = np.empty(k, dtype=object)
    for i in range(k):
        clusters[i] = np.asarray([])
    for x in X:
        x_np = np.asarray(x)
        min = sys.maxsize
        minIndex = 0
        for i in range(k):
            dist = subtract(x_np, centers[i])
            dist_norm = np.linalg.norm(dist)
            if dist_norm < min:
                min = dist_norm
                minIndex = i
        np.append(clusters[minIndex], x_np)
    return clusters


def sum2(cluster):
    a = d
    b = np.size(np.asarray(cluster))
    result = np.zeros(d)
    for i in range(len(cluster)):
        for j in range(d):
            result[j] = result[j] + cluster[i][j]
    return result


def calcCenters(X, clusters):
    centers = np.empty(k, dtype=object)
    for i in range(k):
        sum = sum2(clusters[i])  # np.sum(clusters[i])
        centers.append(sum / np.size(clusters[i]))
    return centers


def kmeans(X):
    centers = initCenters()
    clusters = np.empty(k, dtype=object)
    for i in range(k):
        clusters[i] = np.asarray([])
    converged = False
    while not converged:
        prevCenters = centers
        clusters = calcClusters(X, centers)
        centers = calcCenters(X, clusters)
        converged = isConverged(prevCenters, centers)
    return clusters


def normalize(x):
    return [number / 2 for number in x]


def main():
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    trainSampleSize = 10  # len(x_train)
    X = np.empty(trainSampleSize, dtype=object)
    for i in range(0, trainSampleSize):
        r = random.randint(1, 60000)
        c = x_train[r]
        X[i] = normalize(x_train[r])
        a = normalize(x_train[r])
        b = 5
    clusters = kmeans(X)
    print(clusters)


def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize=15);
        index += 1
    plt.show()


if __name__ == '__main__':
    main()


