# Logistic Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

d1 = unpickle('data_batch_1')
d2 = unpickle('data_batch_2')
d3 = unpickle('data_batch_3')
d4 = unpickle('data_batch_4')
d5 = unpickle('data_batch_5')
t = unpickle('test_batch')

labels1 = d1[b'labels']
labels2 = d2[b'labels']
labels3 = d3[b'labels']
labels4 = d4[b'labels']
labels5 = d5[b'labels']
y_test = t[b'labels']

data1 = d1[b'data']
data2 = d2[b'data']
data3 = d3[b'data']
data4 = d4[b'data']
data5 = d5[b'data']
X_test = t[b'data']

X_train = np.concatenate((data1, data2, data3, data4, data5), axis=0)
y_train = np.array(labels1 + labels2 + labels3 + labels4 + labels5)

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

del labels1, labels2, labels3, labels4, labels5, data1, data2, data3, data4, data5

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
clf_one_x_all = LogisticRegression(solver = 'liblinear', multi_class= 'ovr', random_state = 0)
clf_multinom = LogisticRegression(solver = 'sag', multi_class= 'multinomial', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)

# Importingthe Keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the hidden layer
classifier.add(Dense(output_dim = 1500, activation = 'relu', init='uniform', input_dim = 3072)) # 6 neurons it's the mean of input layer and output layer

# Adding the second hidden layer
classifier.add(Dense(output_dim = 1500, init = 'uniform', activation = 'relu')) # not necessary to inform the input dimension

# Adding the output layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'softmax')) # One output, using sigmoid for propabilistic outcome

# Compiling the ANN
# adam = a type of stochastic gradient descent
# loss = crossentropy for sigmoid function
# metrics = tunned for best accuracy
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 200, epochs = 100)

y_pred = classifier.predict(X_test)
y_pred = lb.inverse_transform(y_pred)
y_test = lb.inverse_transform(y_test)