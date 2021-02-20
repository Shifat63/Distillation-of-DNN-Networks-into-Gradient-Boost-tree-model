from numpy import vstack
import numpy as np
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 26)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()

        # second hidden layer
        self.hidden2 = Linear(26, 13)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()

        # Otput layer
        self.output = Linear(13, 1)
        xavier_uniform_(self.output.weight)
        self.act3 = Sigmoid()

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)

        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)

        # Output layer
        X = self.output(X)
        X = self.act3(X)
        return X
    
    def forwardLastHiddenLayer(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        return X
    
# train the model
def train_model(train_dl, test_dl, model):
    # define the optimization
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9)
    iters  = [] # save the iteration counts here for plotting
    losses = [] # save the avg loss here for plotting
    vallosses = [] # save the avg loss here for plotting
    # enumerate epochs
    for epoch in range(50):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            curr_loss = 0
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            curr_loss += loss
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
        iters.append(epoch)
        losses.append(float(curr_loss/ len(train_dl.dataset)))
        for i, (inputs, targets) in enumerate(test_dl):
            curr_loss = 0
            yhat = model(inputs)
            loss = criterion(yhat, targets.float())
            curr_loss+=loss
        vallosses.append(float(curr_loss/len(test_dl.dataset)))
     #after calculating error per epoch
      
    plt.plot(iters, losses, "r")
    plt.plot(iters, vallosses, "b")
    plt.title("Training Curve (batch_size=1, lr=0.005)")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

# Generate soft labels
def get_soft_labels(data, model):
    xinputs, predictions, true = [], [], []
    for i, (inputs, targets) in enumerate(data):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        predictions.append(np.asscalar(yhat))
        true.append(targets.item())
        xinputs.append(inputs.numpy().flatten())
    return np.array(xinputs), predictions, true

# xinputs: get activations from tha last hidden layer and put them into the xinputs which will be inputs for training the regression models
# oinputs: contain original inputs
# true: the true labels
def get_last_layer(data, model):
    xinputs, true, oinputs = [], [], []
    for i, (inputs, targets) in enumerate(data):
        yhat = model.forwardLastHiddenLayer(inputs)
        yhat = yhat.detach().numpy()
        xinputs.append(yhat.flatten())
        oinputs.append(inputs.numpy().flatten())
        true.append(targets.item())
    return np.array(xinputs), np.array(true), np.array(oinputs)

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        # round to class values
        yhat = yhat.round()
        predictions.append(yhat)
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    acc = accuracy_score(actuals, predictions)
    return acc



