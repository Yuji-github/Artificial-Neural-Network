# Artificial Neural Networks

import numpy as np
import pandas as pd

# tensorflow:
# must change python interpreter -> python 3.8 (tf) for tensorflow, otherwise this can't work
# ignore problems if you changed the env
import tensorflow as tf

# splitting dataset
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

def ann():
    # import dataset
    dataset = pd.read_csv('Churn_Modelling.csv')

    # checking the dataset
    # pd.set_option('display.max_columns', len(dataset.columns))
    # #set_option('rows/columns', max of length for rows/columns)
    # print(dataset.head(1))

    '''
       RowNumber  CustomerId   Surname  CreditScore Geography  Gender  Age   Tenure  Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary   Exited
    0          1    15634602  Hargrave          619    France  Female   42        2      0.0              1          1               1        101348.88        1
    '''

    # transform categorical to numerical
    # (before separate the dataset to in/dependent, this must be done. otherwise, it will be errors in trai_test_split())
    gender = {'Male': 0, 'Female': 1}
    dataset.Gender = [gender[item] for item in dataset.Gender]

    country = {'France': 1, 'Germany': 2, 'Spain': 3}
    dataset.Geography = [country[item] for item in dataset.Geography]
    # print(dataset.head(1))

    '''
          RowNumber  CustomerId   Surname  CreditScore Geography  Gender  Age   Tenure  Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary   Exited
       0          1    15634602  Hargrave          619         1       1   42        2      0.0              1          1               1        101348.88        1
    '''

    # independent
    independent = dataset.iloc[:, 3:-1].values  # store as numpy, and remove the first 3 columns as they are not related
    # len_ind = dataset.iloc[:, 3:-1] # store as pandas
    # print(independent)

    # dependent
    dependent = dataset.iloc[:, -1].values  # store as numpy
    # len_de = dataset.iloc[:, -1] # store as pandas

    # split dataset into 4 parts
    x_train, x_test, y_train, y_test = train_test_split(independent, dependent, train_size=0.8, random_state=0)

    # feature scaling: must do feature scaling for deep learning
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test) # transform() is because checking x_train fit_transform

    # print(x_train)
    # Building Artificial Neural Networks with tensorflow

    # initializing ANN
    ann = tf.keras.models.Sequential() #  Sequential provides training and inference features on this model.

    # adding input layers into the first hidden layers
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

    # Dense(): Just your regular densely-connected NN layer
        # units: Positive integer, dimensionality of the output space.
        # how many neurons you want to have? -> Need to calculate on your own, in this case, already know 6 as education purposes

        # Activation function: If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
        # relu: rectified linear unit activation function.
        # With default values, this returns the standard ReLU activation: max(x, 0), the element-wise maximum of 0 and the input tensor.

    # adding the second layers !adding hidden layers are more like copy and paste
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

    # adding output layer
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    # units=1 because the dependent variable has 0 or 1. so it just needs 1D
    # if the dependent variable has 'a', 'b', 'c' (no relations), it would be 3 as they are 3D
    # like 'a' = 0,0,1   'b' = 0,1,0    'c' =1,0,0
    # sigmoid returns as 0 or 1

    # train/fit ANN

    # compiling ANN
    # Before you can call fit(), you need to specify an optimizer and a loss function
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # optimizer='adam': stochastic gradient descent for training deep learning models
    # loss='binary_crossentropy: for binary outcomes and computes the cross-entropy loss between true labels and predicted labels
    # metrics=['accuracy']: Calculates how often predictions equal labels

    # training ANN
    ann.fit(x_train, y_train, batch_size=32, epochs=100)
    # batch_size=32 (usually): batch size < number of all samples
    # epochs: one forward pass and one backward pass of all the training examples

    # prediction
    # y_pred = ann.predict(x_test) # regular pred with x_test
    
    pred = sc.transform([[600, 1, 0, 40, 3, 6000, 2, 1, 1, 50000]]) # need to transform as input values were transformed
    print(pred)

    # pred returns as prediction
    if(ann.predict(pred) > 0.5):
        print("\nThis guy will leave the bank in six months")
        print(ann.predict(pred))

    else:
        print('\nThis guy will NOT leave the bank in six months')
        print(ann.predict(pred))


if __name__ == '__main__':
    ann()
