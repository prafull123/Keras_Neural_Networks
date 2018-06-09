import pandas as pd
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils


# TEST AND TRAIN DATA SPLIT
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Visualising the Data 
plt.subplot(221)
# First image 
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
# Second Image 
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.show()



seed = 7
numpy.random.seed(seed)


# 28* 28 = 784 Pixels in total :)
num_pixels = X_train.shape[1] * X_train.shape[2]

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')


# normalize inputs from 0-255 to 0-1 (BINARSING THE DATA)
X_train = X_train / 255
X_test = X_test / 255


# One hot Encoding /Binary Feature for Categorical Labels WHICH IS 10 class
# means 10 columns ..of binary values// hot encoded

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# Dense Neural Network instead of Convolutional Neural Network

def Dense_NN_Model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


    
# build the model
model = Dense_NN_Model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model

scores = model.evaluate(X_test, y_test, verbose=0)
print("Dense NN Error: %.2f%%" % (100-scores[1]*100))


Predicted_Output = model.predict(X_test)

print ("PREDICTED_OUTPUT_Probabilities :(One hot encoding format)")
df = pd.DataFrame(Predicted_Output)
print (df)


