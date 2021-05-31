import random
import matplotlib.pyplot as plt
import mnist
from Kmeans import Kmeans

k = 10
d = 28 * 28  # 784


def main():
    # mnist.init()
    x_train, y_train, x_test, y_test = mnist.load()
    x_train = x_train / 255
    x_test = x_test / 255
    # showImages(x_train, y_train)
    kmeans = Kmeans(k, d, x_train, y_train)
    kmeans.cluster(init='random')
    # kmeans.cluster(init='known')

    print("got clusters")
    corrects = kmeans.predict(x_test, y_test)
    # kmeans.showClusters()
    print(corrects)


def showImages(X, Y):
    imgs = []
    labels = []
    amount = 10
    for i in range(0, amount):
        r = random.randint(1, X.shape[0]) - 1
        img = X[r, :].reshape(28, 28)
        imgs.append(img)
        labels.append('training image [' + str(r) + '] = ' + str(Y[r]))

    show_images(imgs, labels)


def show_images(images, title_texts):
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
    plt.show()


if __name__ == '__main__':
    main()
