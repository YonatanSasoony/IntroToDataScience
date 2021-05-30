import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import mnist

k = 10
d = 28 * 28  # 784


def normalize(x):
    return [number / 2 for number in x]


def main():
    # mnist.init()
    x_train, y_train, x_test, y_test = mnist.load()
    x_train = x_train / 255
    x_test = x_test / 255

    idx = np.random.randint(60000, size=1000)
    clusters = kmeans(x_train[idx, :])

    display_cluster(clusters, 0)


def isConverged(prevCenters, centers):
    return np.array_equal(prevCenters, centers)


def kmeans(X):
    centers = initCenters()
    clusters = emptyClusters()  # list of np arrays - list of matrix m,d - m number of vectors in cluster

    converged = False
    while not converged:
        prevCenters = centers
        clusters = calcClusters(X, centers)  # TODO fix- if there is an empty cluster- everything fails
        centers = calcCenters(clusters)
        converged = isConverged(prevCenters, centers)
    return clusters


def initCenters():
    centers = np.empty((0, d), dtype=float)
    for i in range(k):
        centers = np.append(centers, np.random.rand(1, d), axis=0)
    return centers


def calcCenters(clusters):
    centers = np.empty((0, d), dtype=object)
    for i in range(k):
        # sum = sum2(clusters[i])  # np.sum(clusters[i])
        center = np.mean(clusters[i], axis=0)  # vector d,1
        centers = np.append(centers, center.reshape(1, d), axis=0)

    return centers


def emptyClusters():
    clusters = []
    for i in range(k):
        clusters.append(np.empty((0, d), dtype=float))

    return clusters


def calcClusters(X, centers):
    clusters = emptyClusters()  # list of np arrays - list of matrix m,d - m number of vectors in cluster
    for x in X:
        minNorm = sys.maxsize
        minIndex = 0
        for i in range(k):
            dist = np.subtract(x, centers[i])  # vector d,1
            dist_norm = np.linalg.norm(dist)  # scalar
            if dist_norm < minNorm:
                minNorm = dist_norm
                minIndex = i
        clusters[minIndex] = np.append(clusters[minIndex], x.reshape(1, d), axis=0)

    return clusters


def display_cluster(clusters, index):
    imgs = []
    for i in range(0, 10):
        r = random.randint(1, clusters[index].shape[0])
        img = clusters[index][r].reshape(28, 28)
        imgs.append(img)

    show_images(imgs, ['' for img in imgs])


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
        if title_text != '':
            plt.title(title_text, fontsize=15)
        index += 1
    plt.show()


if __name__ == '__main__':
    main()
