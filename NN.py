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
    '''
    Define a class that extends Module class.
    '''
    def __init__(self, n_inputs):
        '''
        The constructor of MLP class defines the layers. We define MLP
        with input layer(n_inputs = 13 neurons), first hidden layer(26 neurons),
        second hidden layer(13 neurons) and output layer. That gives us 689
        weights to be adjusted. We are using Kaiming for the weight initialisation
        strategy for hidden1 - > hidden2, hidden2 - > output layer, according to the
        fact that we are using ReLU() activation function for the both hidden layers.
        For the output layer, we are using Sigmoid() activation function, suitable
        for our binary classification task. We are using Xavier initialization for
        the weights from hidden2 -> output, because it can solve Sigmoid() vanishing
        gradient problem.
        '''
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
        '''
        This function takes the input data (rows of heart.csv, without original
        target labels). The input data is fed in the forward direction through
        the network. Each hidden layer accepts the input data, processes it, as per
        the activation function and passes to the successive layer.

        Parameters
        ----------
        X : input values from heart.csv dataset (without the last "target column")

        Returns
        -------
        X : calculated output(target) value for the given input

        '''
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
        '''
        This function takes the input data (rows of heart.csv, without original
        target labels). The input data is fed in the forward direction through
        the network. Each hidden layer accepts the input data, processes it, as per
        the activation function and passes to the successive layer. This function
        returns the output of the second hidden layer. The goal is to extract the
        learned features from the last hidden layer (in our case - second hidden layer)

        Parameters
        ----------
        X : input values from heart.csv dataset (without the last "target column")

        Returns
        -------
        X : learned features from the second hidden layer

        '''
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        return X
    
# train the model
def train_model(train_dl, test_dl, model):
    '''
    For training of defined MLP model, we have to define loss function
    and optimization algorithm that will be used. Binary cross entropy
    loss is used as loss function. Stochastic gradient descent is used
    for optimization. SGD class provides standard algorithm. In the outer
    loop, we are defining the number of training epochs. In each epoch,
    the inner loop is required for enumerating the mini batches for SGD.
    Each update of the model consists of the following steps: clear the
    gradients, feed the inputs to the network, calculate loss, backpropagate
    the error through the network, update model weights.Additionaly, this
    function plots training and validation learning curves.

    Parameters
    ----------
    train_dl : training dataset
    test_dl : test dataset
    model : object of MLP class

    Returns
    -------
    None.

    '''
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
    '''
    For implementing the first pipeline, we are using predicted soft labels
    for GBT training. This function returns predicted soft labels, along
    with inputs, we keep track on the inputs for which we make predictions,
    along with the target(original) labels.
    Parameters
    ----------
    data : training dataset
    model : object of MLP class
    Returns
    -------
    xinputs : inputs in the order in which we calculate predictions
    predictions : soft labels, without rounding
    true : original (true) target values

    '''
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
    '''
    For implementing the second pipeline, we are using logistic regression
    as helper classifier. We are extracting activations from the last hidden
    layer and feed them into the helper classifier to predict the original
    task. This function returns the activations from the last hidden layer.
    We keep track on the inputs for which we extract activations, along with
    the target(original) labels.

    Parameters
    ----------
    data : training dataset
    model : object of MLP class

    Returns
    -------
    xinputs : activations from the last hidden layer
    oinputs : inputs in the order in which we calculate activations
    true : original (true) target values

    '''
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
    '''
    After we have trained our model, we are calculating the accuracy of
    trained model on the test dataset - percentage of samples that are
    classified correctly.

    Parameters
    ----------
    test_dl : test dataset
    model : object of MLP class

    Returns
    -------
    acc : model accuracy on test dataset

    '''
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



