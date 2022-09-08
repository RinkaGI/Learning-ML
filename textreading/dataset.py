# Import modules
from sklearn.datasets import load_digits
import pandas as pd
import matplotlib.pyplot as plt

mnist = load_digits()

def receiveImage(image: int):
    return mnist.images[image]

def showImage(image: int):
    plt.imshow(mnist.images[image])

def showAllImages():
    _, axes = plt.subplots(nrows=1, ncols=10, figsize=(16, 4))
    for ax, image, label in zip(axes, mnist.images, mnist.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title('Training: %i' % label)

if __name__ == '__main__':
    showAllImages()