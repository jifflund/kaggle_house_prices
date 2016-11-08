'''Classification of the customer support dataset

200M

'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from sklearn.preprocessing import normalize
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K

import csv
import re
from pdb import set_trace

from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer

limit_samples = None
nb_epoch = 3000
train_percentage = 0.8
test_data = 'test.csv'
train_data = 'train.csv'
submission_filename = 'submission_v11.csv'

#
# def tokenize(data):
#     return [x.strip().lower() for x in re.split('(\W+)?', data) if x.strip()]
#     # TODO add in lowercase all words

# use subset or all rows in the data
def load_csv(filename):
    with open(filename, "rb" ) as theFile:
        reader = csv.DictReader( theFile )
        data = []
        for row in reader:
            hsh = {
                'LotArea': int(row['LotArea']),
                'MoSold': int(row['MoSold']),
                'YrSold':  int(row['YrSold']),
                'BedroomAbvGr': int(row['BedroomAbvGr']),
                'Neighborhood': row['Neighborhood'],
                'BldgType': row['BldgType'],
                'OverallQual': int(row['OverallQual']),
                'OverallCond': int(row['OverallCond']),
                'YearBuilt': int(row['YearBuilt']),
                'YearRemodAdd': int(row['YearRemodAdd']),
                'FullBath': int(row['FullBath']),
                'HalfBath': int(row['HalfBath']), #'KitchenQual': int(row['KitchenQual']),
                'GarageType': row['GarageType'],
                'SaleCondition': row['SaleCondition'], #'ExterQual': int(row['ExterQual']),
                'Id': row['Id'],
                'MasVnrType': row['MasVnrType'],
                'LotConfig': row['LotConfig'],
                'Exterior1st': row['Exterior1st'],
                '2ndFlrSF': row['2ndFlrSF'],
                'Utilities': row['Utilities'],
                'Electrical': row['Electrical'],
                'HouseStyle': row['HouseStyle'],
                'SaleType': row['SaleType'],
                'Foundation': row['Foundation'],
                # 'BsmtFullBath': int(row['BsmtFullBath']),
                'HeatingQC': row['HeatingQC'],
                # 'LotFrontage': int(row['LotFrontage']),
                'CentralAir': row['CentralAir'],
                'BsmtExposure': row['BsmtExposure'],
                # 'BsmtFinSF1': row['BsmtFinSF1'],
                'LotShape': row['LotShape'],
                # 'BsmtHalfBath': row['BsmtHalfBath'],
                # 'TotalBsmtSF': int(row['TotalBsmtSF']),
                'TotRmsAbvGrd': int(row['TotRmsAbvGrd']),
                'PavedDrive': row['PavedDrive'],
                # 'GarageYrBlt': int(row['GarageYrBlt']),
                'Exterior2nd': row['Exterior2nd'],
                'Heating': row['Heating'],
                '1stFlrSF': int(row['1stFlrSF']),
                'RoofMatl': row['RoofMatl'],
                'Fireplaces': int(row['Fireplaces']),
                'Functional': row['Functional'],
                'Alley': row['Alley'],
                'RoofStyle': row['RoofStyle'],
                'Street':row['Street'],
                'GarageCars': row['GarageCars']
            }
            if row.get('SalePrice'): hsh['SalePrice'] = int(row['SalePrice'])
            data.append(hsh)
        return data[0:limit_samples]

def get_data(row):
    if row.get('Id'): del row['Id']
    if row.get('SalePrice'): del row['SalePrice']
    return row

train = load_csv(train_data)
Y = [row['SalePrice'] for row in train]
X = [get_data(row) for row in train]
vec = DictVectorizer()
X = vec.fit_transform(X).toarray().tolist()
del train

test = load_csv(test_data)
X_test_label = [row['Id'] for row in test]
X_test = [get_data(row) for row in test]
X_test = vec.transform(X_test).toarray().tolist()
del test


# set_trace()

# randomize the data
rng_state = np.random.get_state()
np.random.shuffle(X)
np.random.set_state(rng_state)
np.random.shuffle(Y)

# Scale the data and turn to categories
min_max_X_scaler = preprocessing.MinMaxScaler()
X = min_max_X_scaler.fit_transform(X)
X_test = min_max_X_scaler.transform(X_test)

min_max_Y_scaler = preprocessing.MinMaxScaler()
Y = min_max_Y_scaler.fit_transform(Y)
# Y = np_utils.to_categorical(Y)

input_normalizer = preprocessing.Normalizer().fit(X)
X = input_normalizer.transform(X)
X_test = input_normalizer.transform(X_test)


maximun_training_sample = int(round(len(X) * train_percentage))

# pick test and validation data
X_train = X[0:maximun_training_sample]
Y_train = Y[0:maximun_training_sample]
X_validate = X[maximun_training_sample:]
Y_validate = Y[maximun_training_sample:]

# set_trace()

model = Sequential()
model.add(Dense(512, input_dim=len(X[0])))
model.add(Activation('relu'))
model.add(Dropout(0.1))
# model.add(addDense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('linear'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['accuracy'])

# set_trace()

# batch_size = len(X_train)
batch_size = len(X_train)

history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_validate, Y_validate))

# score = model.evaluate(X_train, Y_train, verbose=0)
# print("MSE validation: ")
# print("{:,}".format( score[0]) )

# set_trace()

Y_predict = model.predict(X_train)
Y_predict = min_max_Y_scaler.inverse_transform(Y_predict)
Y_train = min_max_Y_scaler.inverse_transform(Y_train.reshape(-1, 1))
print("Train MSE validation: ")
print("{:,}".format( mean_squared_error(Y_predict, Y_train) ))

set_trace()

Y_predict_validate = model.predict(X_validate)
Y_predict_validate = min_max_Y_scaler.inverse_transform(Y_predict_validate)
Y_validate = min_max_Y_scaler.inverse_transform(Y_validate.reshape(-1, 1))
print("Validate MSE validation: ")
print("{:,}".format( mean_squared_error(Y_predict_validate, Y_validate) ))

Y_test = model.predict(X_test)
Y_test = min_max_Y_scaler.inverse_transform(Y_test.reshape(-1, 1))

submission = []
for id, predict in zip(X_test_label, Y_test):
    submission.append([id, predict[0]])


with open(submission_filename, 'wb') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["Id", "SalePrice"])
    for row in submission:
        writer.writerow(row)



