import numpy as np
import matplotlib.pyplot as plt
import dataGetter
import tensorflow as tf





def getData():
    filenames = dataGetter.getFiles()
    print("Getting training images...")
    trainImg = dataGetter.getData(filenames[0,0])
    print("Getting training targets...")
    trainTarg = dataGetter.getData(filenames[0,1])
    print("Getting testing images...")
    testImg = dataGetter.getData(filenames[1,0])
    print("Getting testing targets...")
    testTarg = dataGetter.getData(filenames[1,1])
    print("Data retrieved")
    return trainImg, trainTarg, testImg, testTarg

X, T, Xtest, Ttest = getData()

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(input_shape=(28,28,1),filters=10,kernel_size=12,activation=tf.nn.relu),
  tf.keras.layers.Conv2D(filters=20,kernel_size=8,activation=tf.nn.relu),
  tf.keras.layers.MaxPool2D(pool_size=(2,2),padding="valid"),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, T, epochs=5)
model.evaluate(Xtest, Ttest)


