import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_predict                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
from scipy.stats import randint as sp_randint

def training(classifier, X_train, y_train, X_test, y_test):
    """
    Function that takes the classifier, fits and predicts for test set, returning some metrics
    Parameters:
    -----------
        classifier: the linear model created
        X_train: Training set
        y_train: target variable for training set
        X_test: Test set
        y_test: target variable for test set
    Returns:
    --------
        classifier: the fitted classifier
        metrics: classification metrics
        preds: predictions on training and test set
    """
        
    # Fitting and predicting for training set
    classifier.fit(X_train, y_train)
    y_pred_train = classifier.predict(X_train)
    
    # Getting metrics for training set
    acc_train = accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train, average = 'micro')
    recall_train = recall_score(y_train, y_pred_train, average = 'micro')
    f1_train = f1_score(y_train, y_pred_train, average = 'micro')
    
    # Predicting with cross validation on test set
    y_pred_test = cross_val_predict(estimator = classifier, X = X_test, y = y_test, cv = 5, n_jobs = 3)
    
    # Getting metrics for test set
    acc_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test, average = 'micro')
    recall_test = recall_score(y_test, y_pred_test, average = 'micro')
    f1_test = f1_score(y_test, y_pred_test, average = 'micro')
    cm_test = confusion_matrix(y_test, y_pred_test)
    
    # Saving metrics into a dictionary
    metrics = {'precision_train': precision_train,
               'precision_test':precision_test,
               'f1_score_train':f1_train,
               'f1_score_test': f1_test,
               'accuracy_train': acc_train,
               'accuracy_test': acc_test,
               'recall_train': recall_train,
               'recall_test': recall_test,
               'cm': cm_test}
    
    preds = {'y_pred_train': y_pred_train,
             'y_pred_test': y_pred_test}
    return classifier, preds, metrics

def get_gradients(model):
    """Return the gradient of every trainable weight in model

    Parameters
    -----------
    model : a keras model instance

    First, find all tensors which are trainable in the model. Surprisingly,
    `model.trainable_weights` will return tensors for which
    trainable=False has been set on their layer (last time I checked), hence the extra check.
    Next, get the gradients of the loss with respect to the weights.

    """
    weights = [tensor for tensor in model.trainable_weights if model.get_layer(tensor.name[:-2]).trainable]
    optimizer = model.optimizer

    return optimizer.get_gradients(model.total_loss, weights)

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
y_test = np.array(t[b'labels'])

data1 = d1[b'data']
data2 = d2[b'data']
data3 = d3[b'data']
data4 = d4[b'data']
data5 = d5[b'data']
X_test = t[b'data']

X_train = np.concatenate((data1, data2, data3, data4, data5), axis=0)
y_train = np.array(labels1 + labels2 + labels3 + labels4 + labels5)

del labels1, labels2, labels3, labels4, labels5, data1, data2, data3, data4, data5

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
# Getting F and P Values
from sklearn.feature_selection import f_regression
F, pval = f_regression(X_train, y_train)

# Excluding features with low P value
X_train = X_train[:, pval > 0.05]
X_test = X_test[:, pval > 0.05 of .csv files and outputs an averaged submission. ]
"""
# Fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression

# Model 1 - Logistic Regression One vs All
clf_one_x_all = LogisticRegression(solver = 'sag', multi_class= 'ovr', random_state = 0, C=1000)
clf_one_x_all, preds_one_x_all, metrics_one_x_all2 = training(clf_one_x_all, X_train, y_train, X_test, y_test)

# Model 2 - Logistic Regression Multinomial
clf_multinom = LogisticRegression(solver = 'sag', multi_class= 'multinomial', random_state = 0, C=0.001, n_jobs=2)
clf_multinom, preds_multinom, metrics_multinom = training(clf_multinom, X_train, y_train, X_test, y_test)

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
parameters = {'C': [1,10,100,1000]                                                                              , 
               'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
               }
               
             
random_search = RandomizedSearchCV(estimator = clf_one_x_all,
                           param_distributions = parameters,
                           scoring = 'accuracy',
                           n_iter = 10,
                           n_jobs = 4)
random_search = random_search.fit(X_train, y_train)
best_accuracy = random_search.best_score_
best_parameters = random_search.best_params_

# Model 3 - Neural Nets
# Importingthe Keras library and packages
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Initialising the ANN and adding the layers
clf_nn = Sequential()
clf_nn.add(Dense(output_dim = 500, activation = 'relu', init='uniform', input_dim = 3072)) 
clf_nn.add(Dropout(0.4))
clf_nn.add(Dense(output_dim = 500,  actiitvation = 'relu')) # not necessary to inform the input dimension
clf_nn.add(Dropout(0.4))
clf_nn.add(Dense(output_dim = 10, activation = 'softmax')) # One output, using sigmoid for propabilistic outcome
#clf_nn.add(Dropout(0.3))

# Compiling the ANN
# adam = a type of stochastic gradient descent
# loss = crossentropy for sigmoid function
# metrics = tunned for best accuracy
clf_nn.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
# creating dummies to the neural net
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# Training the neural net
clf_nn.fit(X_train, y_train, batch_size = 400, epochs = 50)

# Getting the numerical gradient
grad = get_gradients(clf_nn)

# Predicting the Test set results
y_pred_nn = clf_nn.predict(X_test)

# Getting metrics, must return to undummed format
y_pred_nn = lb.inverse_transform(y_pred_nn)
y_test = lb.inverse_transform(y_test)
y_train = lb.inverse_transform(y_train)

# Metrics for neural net
acc_test = accuracy_score(y_test, y_pred_nn)
precision_test = precision_score(y_test, y_pred_nn, average = 'micro')
recall_test = recall_score(y_test, y_pred_nn, average = 'micro')
f1_test = f1_score(y_test, y_pred_nn, average = 'micro')
cm_test = confusion_matrix(y_test, y_pred_nn)


import seaborn as sns
sns.heatmap(cm_test, annot=False, square=True)