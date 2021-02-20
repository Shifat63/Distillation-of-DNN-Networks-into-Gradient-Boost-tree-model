import numpy as np
import NN as nn
import GBT as gbt
import GetTrainAndTestData as data
import torch
from numpy import vstack
import pickle

# Getting train and test data from specified csv
from sklearn.linear_model import LogisticRegression

train_dl, test_dl = data.prepare_data('heart.csv')
print('Training ', len(train_dl.dataset))
print('Test ', len(test_dl .dataset))
# print(train_dl)
xTest, yTest = [], []
for i, (inputs, targets) in enumerate(test_dl):
    xTest.append(inputs.numpy().flatten())
    yTest.append(targets.numpy().flatten())
xTrain, yTrain = [], []
for i, (inputs, targets) in enumerate(train_dl):
    xTrain.append(inputs.numpy().flatten())
    yTrain.append(targets.item())
xTrain = np.array(xTrain)
yTrain = np.array(yTrain)


# define the NN
model = nn.MLP(13)
# train the NN, and save the trained model
nn.train_model(train_dl, test_dl, model)
torch.save(model.state_dict(), 'trainedNN.pt')
# Load trained model.
model.load_state_dict(torch.load('trainedNN.pt'))
# test the NN
acc = nn.evaluate_model(test_dl, model)
print('NN Accuracy: %.3f' % (acc*100.0))


# START: Implementing first pipeline
# Generate soft labels from NN
xinputs, predictions, true = nn.get_soft_labels(train_dl, model)
# Train GBT on the soft labels obtained from the neural network
gbtModel = gbt.trainXGbtClassification(xinputs, predictions)
# Calculating accuracy for first pipeline
acc = gbt.testGbt(gbtModel, np.array(xTest), yTest)
print('GBT(only soft labels) Accuracy: %.3f' % (acc*100.0))
# Show tree. 15 is the block size
# gbt.showTree(gbtModel, 15)
# END: Implementing first pipeline


# START: Implementing second pipeline
# Get learned features from NN
xinputsLearned, true, oinputs = nn.get_last_layer(train_dl, model)
# Feed helper classifier with obtained features to predict the original task
logisticRegr = LogisticRegression()
logisticRegr.fit(xinputsLearned, true)
# Train GBT on the soft labels obtained from helper classifier
predictions = (logisticRegr.predict_proba(xinputsLearned))[:, 1]
gbtModel = gbt.trainXGbtClassification(oinputs, predictions)
# Calculating accuracy for second pipeline
acc = gbt.testGbt(gbtModel, np.array(xTest), yTest)
print('GBT(with helper classifier) Accuracy: %.3f' % (acc * 100.0))
# Show tree. 15 is the block size
# gbt.showTree(gbtModel, 15)
# END: Implementing second pipeline


# START: Implementing with hard labels
# Train GBT on the hard labels
gbtModel = gbt.trainXGbtClassification(xTrain, yTrain)
# Calculating accuracy for hard labels
acc = gbt.testGbt(gbtModel, np.array(xTest), yTest)
print('GBT(hard labels) Accuracy: %.3f' % (acc*100.0))
# Show tree. 15 is the block size
# gbt.showTree(gbtModel, 15)
# END: Implementing with hard labels





