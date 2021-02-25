import numpy as np
import NN as nn
import GBT as gbt
import GetTrainAndTestData as data
import torch
from sklearn.linear_model import LogisticRegression


def trainAndSaveNN(train_dl, test_dl, model):
    '''
    Train the Neural Network and save the trained model

    Parameters
    ----------
    train_dl : Training data
    test_dl : Test data
    model : Neural Network model

    Returns
    -------
    None.

    '''
    nn.train_model(train_dl, test_dl, model)
    torch.save(model.state_dict(), 'trainedNN.pt')


def firstPipeline(train_dl, model, xTest, yTest):
    '''
    Implements the first pipeline. Steps are
    1. Get soft labels from trained NN model
    2. Train GBT with the soft labels

    Finally it calculates the accuracy of trained GBT model
    with respect to test data and plot decision trees.

    Parameters
    ----------
    train_dl : Training data
    model : Trained Neural Network model
    xTest : Test inputs
    yTest : Desired(actual) outputs for test inputs

    Returns
    -------
    None.

    '''
    # Generate soft labels from NN
    xinputs, predictions, true = nn.get_soft_labels(train_dl, model)
    # Train GBT on the soft labels obtained from the neural network
    gbtModel = gbt.trainXGbtClassification(xinputs, predictions)
    # Calculating accuracy
    acc = gbt.testGbt(gbtModel, np.array(xTest), yTest)
    print('GBT(only soft labels) Accuracy: %.3f' % (acc * 100.0))
    # Show tree. 15 is the block size
    # gbt.showTree(gbtModel, 15, 'Pipeline 1')


def secondPipeline(train_dl, model, xTest, yTest):
    '''
    Implements the second pipeline. Steps are
    1. Get learned features from trained NN model
    2. Feed the learned features to the Helper classifier
    3. Train GBT with the soft labels obtained from helper classifier

    Finally it calculates the accuracy of trained GBT model
    with respect to test data and plot decision trees.

    Parameters
    ----------
    train_dl : Training data
    model : Trained Neural Network model
    xTest : Test inputs
    yTest : Desired(actual) outputs for test inputs

    Returns
    -------
    None.

    '''
    # Get learned features from NN
    xinputsLearned, true, oinputs = nn.get_last_layer(train_dl, model)
    # Feed helper classifier with obtained features to predict the original task
    logisticRegr = LogisticRegression()
    logisticRegr.fit(xinputsLearned, true)
    # Train GBT on the soft labels obtained from helper classifier
    predictions = (logisticRegr.predict_proba(xinputsLearned))[:, 1]
    gbtModel = gbt.trainXGbtClassification(oinputs, predictions)
    # Calculating accuracy
    acc = gbt.testGbt(gbtModel, np.array(xTest), yTest)
    print('GBT(with helper classifier) Accuracy: %.3f' % (acc * 100.0))
    # Show tree. 15 is the block size
    # gbt.showTree(gbtModel, 15, 'Pipeline 2')


def gbtWithHardLabels(xTrain, yTrain, xTest, yTest):
    '''
    Train GBT with Hard labels, calculate it's accuracy with test data and plot decision trees.

    Parameters
    ----------
    xTrain : Training inputs
    yTrain : Training outputs (Teacher value)
    xTest : Test inputs
    yTest : Desired(actual) outputs for test inputs

    Returns
    -------
    None.

    '''
    # Train GBT on the hard labels
    gbtModel = gbt.trainXGbtClassification(xTrain, yTrain)
    # Calculating accuracy
    acc = gbt.testGbt(gbtModel, np.array(xTest), yTest)
    print('GBT(hard labels) Accuracy: %.3f' % (acc * 100.0))
    # Show tree. 15 is the block size
    # gbt.showTree(gbtModel, 15, 'GBT(Trained with hard labels)')


# Getting train and test data from specified csv
train_dl, test_dl = data.prepare_data('heart.csv')
print('Training ', len(train_dl.dataset))
print('Test ', len(test_dl.dataset))
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
# Train NN and save the trained model for future use.
# trainAndSaveNN(train_dl, test_dl, model)
# Load trained model.
model.load_state_dict(torch.load('trainedNN.pt'))
# test the NN
acc = nn.evaluate_model(test_dl, model)
print('NN Accuracy: %.3f' % (acc * 100.0))

# Calling first pipeline
firstPipeline(train_dl, model, xTest, yTest)
# Calling second pipeline
secondPipeline(train_dl, model, xTest, yTest)
gbtWithHardLabels(xTrain, yTrain, xTest, yTest)
