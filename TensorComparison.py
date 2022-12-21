# Creates tensorflow model to learn how to convert ints to their ascii values
# Alden Bauman
# CS691
import random
import tensorflow as tf

# initializes input and output keys
# will be using single digit acii number classification generated randomly const
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
numKey = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]
trainSamples = 1000
testSamples = 500

trainX = []
trainY = []
testX = []
testY = []

# generate train and test sets of desired size
for i in range(trainSamples):
    trainX.append(random.randint(0, 9))
    trainY.append(numKey[trainX[i]])
    # print(trainX[i], trainY[i])

for i in range(testSamples):
    testX.append(random.randint(0, 9))
    testY.append(numKey[testX[i]])
    # print(testX[i], testY[i])

# initialize network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dense(60)
])

# compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train network
model.fit(trainX, trainY, epochs=100)

# check performance
test_loss, test_acc = model.evaluate(testX,  testY)

print('\nTest accuracy:', test_acc)

