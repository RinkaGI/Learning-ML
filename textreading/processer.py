import pandas as pd

def flattenImage(mnist):
    nSamples = len(mnist.images)
    data = mnist.images.reshape((nSamples, -1))
    return data

def splitData(mnist, data):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        data, mnist.target, test_size=0.5, shuffle=False
    )
    return X_train, X_test, y_train, y_test