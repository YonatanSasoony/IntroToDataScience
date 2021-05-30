import datetime
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
    start = datetime.datetime.now()
    x_train, y_train, x_test, y_test = mnist.load()
    x_train = x_train / 255
    x_test = x_test / 255

    for i in range(1, 11):
        size = i * 1000
        idx = np.random.randint(60000, size=size)
        clusters_x, clusters_y = kmeans(x_train[idx, :], y_train[idx])
        print("size:"+str(size) + "- " + str(datetime.datetime.now() - start))

    # for i in range(k):
    #     display_cluster(clusters_x, clusters_y, i)


def isConverged(prevCenters, centers):
    return np.array_equal(prevCenters, centers)


def isValidClusters(clusters):
    for i in range(k):
        if np.size(clusters[i]) == 0:
            return False
    return True


def kmeans(X, Y):
    validClusters = False
    while not validClusters:
        centers = initCenters()
        clusters_X, clusters_Y = calcClusters(X, Y, centers)  # list of np arrays - list of matrix m,d - m number of vectors in cluster
        validClusters = isValidClusters(clusters_X)

    converged = False
    while not converged:
        prevCenters = centers
        clusters_X, clusters_Y = calcClusters(X, Y, centers)  # TODO fix- if there is an empty cluster- everything fails
        centers = calcCenters(clusters_X)
        converged = isConverged(prevCenters, centers)
    return clusters_X, clusters_Y


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


def calcClusters(X, Y, centers):
    clusters_X, clusters_Y = emptyClusters()  # list of np arrays - list of matrix m,d - m number of vectors in cluster
    for index, x in enumerate(X):
        minNorm = sys.maxsize
        minIndex = 0
        for i in range(k):
            dist = np.subtract(x, centers[i])  # vector d,1
            dist_norm = np.linalg.norm(dist)  # scalar
            if dist_norm < minNorm:
                minNorm = dist_norm
                minIndex = i
        clusters_X[minIndex] = np.append(clusters_X[minIndex], x.reshape(1, d), axis=0)
        clusters_Y[minIndex].append(Y[index])

    return clusters_X, clusters_Y


def emptyClusters():
    clusters_X, clusters_Y = [], []
    for i in range(k):
        clusters_X.append(np.empty((0, d), dtype=float))
        clusters_Y.append([])

    return clusters_X, clusters_Y


def calcGuessLabel(Y):
    histo = np.zeros(k)
    for i in range(len(Y)):
        histo[Y[i]] += 1
    y = np.argmax(histo)
    if np.size(y) > 1:
        return y[0]
    return y


def display_cluster(clusters_x, clusters_y, index):
    imgs = []
    labels = []
    amount = 10
    nonlabels = ['' for i in range(amount)]
    guessLabel = calcGuessLabel(clusters_y[index])
    for i in range(0, amount):
        r = random.randint(1, clusters_x[index].shape[0]) - 1
        img = clusters_x[index][r].reshape(28, 28)
        imgs.append(img)
        labels.append('training image [' + str(r) + '] = ' + str(clusters_y[index][r]))
    show_images(imgs, nonlabels, str(guessLabel))


def show_images(images, title_texts, main_title):
    cols = 5
    rows = int(len(images) / cols)
    plt.figure(figsize=(20, 10))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize=15)
        index += 1
    print(main_title)
    plt.show()


if __name__ == '__main__':
    main()
