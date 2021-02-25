from xgboost import XGBClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

objective = "multi:softprob" #Used for multiclass classification
maxNumberOfTrees = 100
learningRate = 0.1
maxTreeDepth = 3

#Train the model
def trainXGbtClassification(X, y):
    '''
    objective function = "multi:softprob" (Used for multiclass classification)
    Maximum number of trees = 100
    Learning rate = 0.1
    Maximum tree depth = 3

    For first pipeline:
    We pass the training inputs and generated soft labels from the NN to
    this function to train the Gradient Boosting Tree(GBT).
    For second pipeline:
    We pass the training inputs and generated soft labels from the Helper classifier to
    this function to train the Gradient Boosting Tree(GBT).

    Parameters
    ----------
    X : Training inputs
    y : Soft labels

    Returns
    -------
    model : Trained GBT model

    '''
    model = XGBClassifier(learning_rate=learningRate, n_estimators=maxNumberOfTrees, max_depth=maxTreeDepth)
    model.fit(X, y)
    return model

#Test accuracy of the model
def testGbt(model, X, y):
    '''
    After we have trained our GBT model, we are calculating the accuracy of
    trained model on the test dataset - percentage of samples that are
    classified correctly.

    Parameters
    ----------
    model : trained GBT model
    X : Test inputs
    y : Desired(actual) output for the given test inputs

    Returns
    -------
    acc : model accuracy on test dataset

    '''
    predictions = model.predict(X)
    for i in range(len(predictions)):
        predictions[i] = round(predictions[i])
    acc = accuracy_score(y, predictions)
    return acc

#Show GBT trees. Last one from each block
def showTree(model, blockSize, title):
    '''
    Show GBT trees. Last tree from each block.
    To save computational power all the trees generated are not shown.
    If it is required to show all trees then provide blockSize=1

    Parameters
    ----------
    model : trained GBT model
    blockSize : size of block from which last tree will be shown
    title : Title of the figure

    Returns
    -------
    None.

    '''
    for i in range(int(maxNumberOfTrees/blockSize)):
        ax = plot_tree(model, num_trees=i*blockSize+(blockSize-1))
        ax.title.set_text(title + ', Tree number: ' + str(i*blockSize+blockSize))
        plt.show()