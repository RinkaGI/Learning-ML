from model import Model
from dataset import mnist
from processer import *
from PIL import Image

model = Model()

dataset = mnist
data = flattenImage(dataset)

X_train, X_test, y_train, y_test = splitData(dataset, data)

print('Training...')
model.train(X_train, y_train)
print('Loading...')

image = input('Input a image: ')

imageLoaded = Image.open(image)
imageGray = imageLoaded.convert('L')
imageResized = imageGray.resize((64, 64))

print(f'The char detected on your image is: {str(model.predict(imageResized))}')