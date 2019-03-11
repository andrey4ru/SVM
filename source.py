import SVM
import ROC
import pandas as pd

# ------------------------------------- Data preparation -------------------------------------
data = pd.read_csv('iris.csv')  # read data from file

# ------------------------------------- Setosa vs Versicolor ----------------------------------
dataTrain = data[10:50]  # split data on train set and test set
dataTrain = dataTrain.append(data[50:90], ignore_index=True)
label = dataTrain.Species
label = label.replace(['setosa', 'versicolor'], ['1', '-1'])  # replace setosa, versicolor with '-1', '1'
dataTrain.drop(['Species'], axis=1, inplace=True)

dataTest = data[0:10]
dataTest = dataTest.append(data[90:100], ignore_index=True)

target = dataTest.Species
target = target.replace(['setosa', 'versicolor'], ['1', '-1'])  # replace setosa, versicolor with '-1', '1'
dataTest.drop(['Species'], axis=1, inplace=True)

model = SVM.SVM()  # initialize model
model.train(dataTrain, label)  # train model
predict = model.predict(dataTest)  # get predict
ROC.ROC(predict, target, '-1')  # plot ROC curve

# ----------------------------------- Versicolor vs Virginica  -----------------------------------
import SVM
import ROC
import pandas as pd

# ------------------------------------- Data preparation -------------------------------------
data = pd.read_csv('iris.csv')  # read data from file

# ------------------------------------- Setosa vs Versicolor ----------------------------------
dataTrain = data[10:50]  # split data on train set and test set
dataTrain = dataTrain.append(data[50:90], ignore_index=True)
label = dataTrain.Species
label = label.replace(['setosa', 'versicolor'], ['1', '-1'])  # replace setosa, versicolor with '1', '-1'
dataTrain.drop(['Species'], axis=1, inplace=True)

dataTest = data[0:10]
dataTest = dataTest.append(data[90:100], ignore_index=True)

target = dataTest.Species
target = target.replace(['setosa', 'versicolor'], ['1', '-1'])  # replace setosa, versicolor with '1', '-1'
dataTest.drop(['Species'], axis=1, inplace=True)

model = SVM.SVM()  # initialize model
model.train(dataTrain, label)  # train model
predict = model.predict(dataTest)  # get predict
ROC.ROC(predict, target, '1')  # plot ROC curve

# ----------------------------------- Versicolor vs Virginica  -----------------------------------
dataTrain = data[100:140]
dataTrain = dataTrain.append(data[50:90], ignore_index=True)
label = dataTrain.Species
label = label.replace(['versicolor', 'virginica'], ['1', '-1'])  # replace versicolor, virginica with '1', '-1'
dataTrain.drop(['Species'], axis=1, inplace=True)

dataTest = data[90:100]
dataTest = dataTest.append(data[140:150], ignore_index=True)

target = dataTest.Species
target = target.replace(['versicolor', 'virginica'], ['1', '-1'])  # replace versicolor, virginica with '1', '-1'
dataTest.drop(['Species'], axis=1, inplace=True)

model = SVM.SVM()  # initialize model
model.train(dataTrain, label)  # train model
predict = model.predict(dataTest)  # get predict
ROC.ROC(predict, target, '1')  # plot ROC curve

# ------------------------------------ Virginica vs Setosa -------------------------------------
dataTrain = data[100:140]
dataTrain = dataTrain.append(data[10:50], ignore_index=True)

label = dataTrain.Species
label = label.replace(['virginica', 'setosa'], ['1', '-1'])  # replace virginica, setosa with '1', '-1'
dataTrain.drop(['Species'], axis=1, inplace=True)

dataTest = data[140:150]
dataTest = dataTest.append(data[0:10], ignore_index=True)

target = dataTest.Species
target = target.replace(['virginica', 'setosa'], ['1', '-1'])  # replace virginica, setosa with '1', '-1'
dataTest.drop(['Species'], axis=1, inplace=True)

model = SVM.SVM()  # initialize model
model.train(dataTrain, label)  # train model
predict = model.predict(dataTest)  # get predict
ROC.ROC(predict, target, '1')  # plot ROC curve