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
    # this function does not work, idk why, this function isnt important, so i dont care
    
    fig, axes = plt.subplots(2, 10, figsize=(16, 6))

    for i in range(20):
        axes[i//10, i %10].imshow(mnist.images[i], cmap='gray');
        axes[i//10, i %10].axis('off')
        axes[i//10, i %10].set_title(f"target: {mnist.target[i]}")
        
    plt.tight_layout()