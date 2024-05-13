import sys
if sys.version_info[0] < 3:
	raise Exception("Python 3 not detected.")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io
import pandas as pd

np.random.seed(23409859)
    
def shuffleMnist(images):
    #load data
    mnist = np.load('./data/mnist-data.npz')
    mnistData = mnist["training_data"]
    mnistLabels = mnist["training_labels"]

    #shuffle indices of the data/labels
    entries = mnistData.shape[0]
    mnistIndices = np.arange(entries)
    np.random.shuffle(mnistIndices)

    #allocate images number of data/labels to validation, rest to training
    mnistValidationIndices = mnistIndices[:images]
    mnistTrainIndices = mnistIndices[images:]

    #make arrays
    mnistValidationData = mnistData[mnistValidationIndices]
    mnistValidatinLabels = mnistLabels[mnistValidationIndices]

    mnistTrainingData = mnistData[mnistTrainIndices]
    mnistTrainingLabels = mnistLabels[mnistTrainIndices]

    #first two return values are images # of allocated data, rest are allocated to last two return values
    return mnistValidationData, mnistValidatinLabels, mnistTrainingData, mnistTrainingLabels

## Question 3a MNIST##
mnistVal, mnistValLabels, mnistTrain, mnistTrainLabels = shuffleMnist(10000)

def shuffleSpamPercent(percent):
    #load data
    spam = np.load('./data/spam-data.npz')
    spamData = spam["training_data"]
    spamLabels = spam["training_labels"]

    #shuffle indices of the data/labels
    entries = spamData.shape[0]
    spamIndices = np.arange(entries)
    np.random.shuffle(spamIndices)

    #allocate images number of data/labels to validation, rest to training
    spamValidationIndices = spamIndices[:int(percent * entries)]
    spamTrainIndices = spamIndices[int(percent * entries):]

    #make arrays
    spamValidationData = spamData[spamValidationIndices]
    spamValidatinLabels = spamLabels[spamValidationIndices]

    spamTrainingData = spamData[spamTrainIndices]
    spamTrainingLabels = spamLabels[spamTrainIndices]

    #first two return values are images # of allocated data, rest are allocated to last two return values
    return spamValidationData, spamValidatinLabels, spamTrainingData, spamTrainingLabels

## Question 3a SPAM ##
spamVal, spamValLabels, spamTrain, spamTrainLabels = shuffleSpamPercent(0.2)


## Question 3b ##
def evaluate(actual, predicted):
    if (len(actual) != len(predicted)) :
        return
    sum = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            sum +=1
    return sum / len(actual)

###################################################################
#   Question 4  #
def subset(data, labels, n):
    indices = np.arange(n)
    np.random.shuffle(indices)

    #allocate images number of data/labels to validation, rest to training
    usedIndices = indices[:n]

    #make arrays
    return data[usedIndices], labels[usedIndices]


# Question 4a #
mnistNumbers = [100,200,500,1000,2000,5000,10000]
trainAccuracy = []
valAccuracy = []
mnistData = np.load('./data/mnist-data.npz')
# mnistVal, mnistValLabels, mnistTrain, mnistTrainLabels as before

for n in mnistNumbers:
    #obtain subset of training data
    trainData, trainLabels = subset(mnistTrain, mnistTrainLabels, n)

    #train model
    model = svm.SVC(kernel = 'linear')
    
    #flatten data
    valData = mnistVal.reshape(mnistVal.shape[0], -1)
    valLabels = mnistValLabels

    trainData = trainData.reshape(trainData.shape[0], -1)

    #train the model
    model.fit(trainData, trainLabels)

    #evaluate model on training data
    trainPred = model.predict(trainData)
    tAccuracy = evaluate(trainLabels, trainPred)
    trainAccuracy.append(tAccuracy)

    #evaluate model on validation data
    valPred = model.predict(valData)
    vAccuracy = evaluate(valLabels, valPred)
    valAccuracy.append(vAccuracy)

    print(f'{n} samples - training accuracy: {tAccuracy}, validation accuracy: {vAccuracy}')

#plot the training and validation accuracies against number of examples
plt.plot(mnistNumbers, trainAccuracy, label = 'Training Accuracy', marker = 'o', linestyle = '-')
plt.plot(mnistNumbers, valAccuracy, label = 'Validation Accuracy', marker = 'o', linestyle = '-')

plt.xlabel('Number of Examples')
plt.ylabel('Accuracy')

plt.title('# of Examples vs. Model Accuracy for MNIST')
plt.legend()

plt.xlim(0, max(mnistNumbers) + 10)
plt.ylim(0, 1.5)
plt.show()

           

# Question 4b #
spamData = np.load('./data/mnist-data.npz')
spamNumbers = [100,200,500,1000,2000, spamTrain.shape[0]]
trainAccuracy = []
valAccuracy = []
#spamVal, spamValLabels, spamTrain, spamTrainLabels as before
    
for n in spamNumbers:
    trainData, trainLabels = subset(spamTrain, spamTrainLabels, n)

    #train model
    model = svm.SVC(kernel = 'linear', C = 100)
    
    #flatten data
    valData = spamVal.reshape(spamVal.shape[0], -1)
    valLabels = spamValLabels

    trainData = trainData.reshape(trainData.shape[0], -1)

    #train the model
    model.fit(trainData, trainLabels)

    #evaluate model on training data
    trainPred = model.predict(trainData)
    tAccuracy = evaluate(trainLabels, trainPred)
    trainAccuracy.append(tAccuracy)

    #evaluate model on validation data
    valPred = model.predict(valData)
    vAccuracy = evaluate(valLabels, valPred)
    valAccuracy.append(vAccuracy)

    print(f'{n} samples - training accuracy: {tAccuracy}, validation accuracy: {vAccuracy}')

#plot the training and validation accuracies against number of examples
plt.plot(spamNumbers, trainAccuracy, label = 'Training Accuracy', marker = 'o', linestyle = '-')
plt.plot(spamNumbers, valAccuracy, label = 'Validation Accuracy', marker = 'o', linestyle = '-')

plt.xlabel('Number of Examples')
plt.ylabel('Accuracy')

plt.title('# of Examples vs. Model Accuracy for SPAM')
plt.legend()

plt.xlim(0, max(spamNumbers) + 10)
plt.ylim(0, 1.1)
plt.show()





## Question 5 ##
cValues = []
trainAccuracy = []
valAccuracy = []
mnistData = np.load('./data/mnist-data.npz')
n = 1 * 10 **-9
# mnistVal, mnistValLabels, mnistTrain, mnistTrainLabels as before
trainData, trainLabels = subset(mnistTrain, mnistTrainLabels, 10000)

while n < 0.1:
    cValues.append(n)
    #obtain subset of training data

    #train model
    model = svm.SVC(kernel = 'linear', C = n)
    
    #flatten data
    valData = mnistVal.reshape(mnistVal.shape[0], -1)
    valLabels = mnistValLabels

    trainData = trainData.reshape(trainData.shape[0], -1)

    #train the model
    model.fit(trainData, trainLabels)

    #evaluate model on training data
    trainPred = model.predict(trainData)
    tAccuracy = evaluate(trainLabels, trainPred)
    trainAccuracy.append(tAccuracy)

    #evaluate model on validation data
    valPred = model.predict(valData)
    vAccuracy = evaluate(valLabels, valPred)
    valAccuracy.append(vAccuracy)

    print(f'C value of {n} - training accuracy: {tAccuracy}, validation accuracy: {vAccuracy}')
    n = n * 8
#optimal value of c was about 5.12 * 10^-7
##  QUESTION 7 ##
## retrain data on n = 5.12 * 10 ^ -7, obtain predictions from test set
## and export to .csv called mnistPredictions.csv
model = svm.SVC(kernel = 'linear', C = 5.12 * 10**-7)

trainData = trainData.reshape(trainData.shape[0], -1)

#train the model
model.fit(trainData, trainLabels)

#evaluate model on test data
testData = mnistData["test_data"]
testData = testData.reshape(testData.shape[0], -1)
testPred = model.predict(testData)

indices = [x for x in range(1, len(testPred) + 1)]
df = pd.DataFrame({'Id': indices, 'Category': testPred})
df.to_csv('mnistPredictions.csv', index = False, sep = ',')



## Question 6 ##
# 5 -fold cross-validation
# spamVal, spamValLabels, spamTrain, spamTrainLabels as before
spam = np.load('./data/spam-data.npz')
spamData = spam["training_data"]
spamLabels = spam["training_labels"]

size = spamData.shape[0]
chunk = size // 5
spamData, spamLabels = subset(spamData, spamLabels, size)
first, firstLabels = spamData[:chunk], spamLabels[:chunk]
second, secondLabels = spamData[chunk:chunk*2], spamLabels[chunk:chunk*2]
third, thirdLabels = spamData[chunk*2:chunk*3], spamLabels[chunk*2:chunk*3]
fourth, fourthLabels = spamData[chunk*3:chunk*4], spamLabels[chunk*3:chunk*4]
fifth, fifthLabels = spamData[chunk*4:], spamLabels[chunk*4:]

dataParts = [first, second, third, fourth, fifth]
labelParts = [firstLabels, secondLabels, thirdLabels, fourthLabels, fifthLabels]

c = 1 * 10 **-5
cValues = []
valAccuracies = []
trainingAccuracies = []
while c < 500:
    cValues.append(c)
    vAccuracies = []
    tAccuracies = []
    for x in range(0, len(dataParts)): 
        valData = dataParts[x]
        valLabels = labelParts[x]
        
        otherData = [x for x in dataParts if x is not valData]
        otherLabels = [x for x in labelParts if x is not valLabels]

        trainData = otherData[0]
        trainLabels = otherLabels[0]
        for x in range(1, len(otherData)):
            trainData = np.concatenate((trainData, otherData[x]))
            trainLabels = np.concatenate((trainLabels, otherLabels[x]))
            #train model
        model = svm.SVC(kernel = 'linear', C = c)
        
        #flatten data
        valData = valData.reshape(valData.shape[0], -1)

        trainData = trainData.reshape(trainData.shape[0], -1)

        #train the model
        model.fit(trainData, trainLabels)

        #evaluate model on training data
        trainPred = model.predict(trainData)
        tAccuracy = evaluate(trainLabels, trainPred)
        tAccuracies.append(tAccuracy)

        #evaluate model on validation data
        valPred = model.predict(valData)
        vAccuracy = evaluate(valLabels, valPred)
        vAccuracies.append(vAccuracy)

    tAccuracy = sum(tAccuracies) / len(tAccuracies)
    vAccuracy = sum(vAccuracies) / len(vAccuracies)
    print(f'C value of {c} - training accuracy: {tAccuracy}, validation accuracy: {vAccuracy}')
    c = c * 8
#optimal solution was about C = 21
##    QUESTION 7 ###
## retrain data on c = 21 and export to .csv called spamPredictions.csv
valAccuracies = []
trainingAccuracies = []
for x in range(0, len(dataParts)): 
    valData = dataParts[x]
    valLabels = labelParts[x]
    
    otherData = [x for x in dataParts if x is not valData]
    otherLabels = [x for x in labelParts if x is not valLabels]

    trainData = otherData[0]
    trainLabels = otherLabels[0]
    for x in range(1, len(otherData)):
        trainData = np.concatenate((trainData, otherData[x]))
        trainLabels = np.concatenate((trainLabels, otherLabels[x]))
        #train model
    model = svm.SVC(kernel = 'rbf', gamma = 'auto', C = 21)
    
    #flatten data
    valData = valData.reshape(valData.shape[0], -1)

    trainData = trainData.reshape(trainData.shape[0], -1)

    #train the model
    model.fit(trainData, trainLabels)
#evaluate model on testing data
testData = spam["test_data"]
testData = testData.reshape(testData.shape[0], -1)
testPred = model.predict(testData)
indices = [x for x in range(1, len(testPred) + 1)]
df = pd.DataFrame({'Id': indices, 'Category': testPred})
df.to_csv('spamPredictions.csv', index = False, sep = ',')