@ -1,12 +1,14 @@
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, linear_model
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt

#### LOAD DATA ####
print("-"*30);print('IMPORTING DATA ');print("-"*30);
data = pd.read_csv('rent_price_dataset_buca.csv')
# print(data)


type_1 = preprocessing.LabelEncoder()
data["floor_type"] = type_1.fit_transform(data["floor_type"])
data["location"] = type_1.fit_transform(data["location"])
@ -25,19 +27,27 @@ x = np.array(data[['location', 'num_rooms', 'gross', 'building_age', 'floor_type
y = np.array(data['price'])
# x = np.array(data.drop(['price']))
# y = np.array(data['price'])

print(x.shape)
print(y.shape)

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size=0.2)



poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(xTrain)
X_test_poly = poly.transform(xTest)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_poly)
X_test = scaler.transform(X_test_poly)  # Only transform test data

print(f'xTrain {xTrain.shape}')
print(f'yTrain {yTrain.shape}')


model = linear_model.LinearRegression()
model.fit(xTrain, yTrain,fit_intercept= False)

model.fit(xTrain, yTrain)

print(f'Cofficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')
@ -49,21 +59,30 @@ print(f'accuracy :{round(accuracy * 100, 3)}%')
test_val = model.predict(xTest)
print(test_val)

plt.figure(figsize=(10, 5))
plt.plot(yTest, label='Actual Values', color='red', linestyle='dashed', marker='o')
plt.plot(test_val, label='Predicted Values', color='blue', linestyle='dashed', marker='x')
plt.legend()
plt.xlabel("Test Samples")
plt.ylabel("Price")
plt.title("Actual vs Predicted Rent Prices")
plt.show()

error = []
for i, testVal in enumerate(test_val):
    error.append(yTest[i]-testVal)
    print(f'Acutal:{yTest[i]} Prediction:{int(testVal)} Error: {int(error[i])}')




#
#
# import pandas as pd
# from sklearn import preprocessing,linear_model
# import numpy as np
# import sklearn
# #### LOAD DATA ####
# print(‘-‘*30);print(“IMPORTING DATA “);print(‘-‘*30);
# data = pd.read_csv(‘houses_to_rent.csv’, sep = ‘,’)
# # print(‘-‘*30);print(“IMPORTING DATA “);print(‘-‘*30);
# data = pd.read_csv('houses_to_rent.csv')
# data = data [[‘city’,’rooms’,’bathroom’, ‘parking spaces’,’fire insurance’,
# ‘furniture’,’rent amount’]]
# print(data.head())
@ -72,41 +91,41 @@ for i, testVal in enumerate(test_val):
# data[‘rent amount’] = data[‘rent amount’].map(lambda i: int(i[2:].replace(‘,’,”)))
# data[‘fire insurance’] = data[‘fire insurance’].map(lambda i: int(i[2:].replace(‘,’,”)))
# le = preprocessing.LabelEncoder()
# data[‘furniture’] = le.fit_transform((data[‘furniture’]))
# data['furniture'] = le.fit_transform((data['furniture']))
# print(data.head())
#
# print(‘-‘*30);print(“CHECKING NULL DATA “);print(‘-‘*30);
# # print(‘-‘*30);print(“CHECKING NULL DATA “);print(‘-‘*30);
# print(data.isnull().sum())
# #data = data.dropna()
# print(‘-‘*30);print(” HEAD “);print(‘-‘*30);
# # print(‘-‘*30);print(” HEAD “);print(‘-‘*30);
# print(data.head())
#
# #### SPLT DATA ####
# print(‘-‘*30);print(” SPLIT DATA “);print(‘-‘*30);
# x = np.array(data.drop([‘rent amount’],1))
# y = np.array(data[‘rent amount’])
# print(‘X’,x.shape)
# print(‘Y’,y.shape)
# # print(‘-‘*30);print(” SPLIT DATA “);print(‘-‘*30);
# x = np.array(data.drop(['rent amount'],1))
# y = np.array(data['rent amount'])
# print('X',x.shape)
# print('Y',y.shape)
# xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x,y,
# test_size=0.2,
# random_state=10)
# print(‘XTrain’,xTrain.shape)
# print(‘XTest’,xTest.shape)
# print('XTrain',xTrain.shape)
# print('XTest',xTest.shape)
#
# #### TRAINING ####
# print(‘-‘*30);print(” TRAINING “);print(‘-‘*30);
# # print(‘-‘*30);print(” TRAINING “);print(‘-‘*30);
# model = linear_model.LinearRegression()
# model.fit(xTrain,yTrain)
# accuracy = model.score(xTest,yTest)
# print(‘Cofficients: ‘,model.coef_)
# print(‘Intercept: ‘, model.intercept_)
# print(‘Accuracy:’,round(accuracy*100,3),’%’)
# print('Cofficients: ',model.coef_)
# print('Intercept: ', model.intercept_)
# print('Accuracy:',round(accuracy*100,3),'%')
#
# #### EVALUATION ####
# print(‘-‘*30);print(” MANUAL TESTING “);print(‘-‘*30);
# # print(‘-‘*30);print(” MANUAL TESTING “);print(‘-‘*30);
# testVals = model.predict(xTest)
# print(testVals.shape)
# error = []
# for i,testVal in enumerate(testVals):
# error.append(yTest[i]-testVal)
# print(f’Acutal:{yTest[i]} Prediction:{int(testVal)} Error: {int(error[i])}’)
#     error.append(yTest[i]-testVal)
#     print(f'Acutal:{yTest[i]} Prediction:{int(testVal)} Error: {int(error[i])}')
