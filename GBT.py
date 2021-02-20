from xgboost import XGBClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

objective = "multi:softprob" #Used for multiclass classification
maxNumberOfTrees = 100
learningRate = 0.1
maxTreeDepth = 3

#Train the model
def trainXGbtClassification(X, y):
    model = XGBClassifier(learning_rate=learningRate, n_estimators=maxNumberOfTrees, max_depth=maxTreeDepth)
    model.fit(X, y)
    return model

#Test accuracy of the model
def testGbt(model, X, y):
    predictions = model.predict(X)
    for i in range(len(predictions)):
        predictions[i] = round(predictions[i])
    acc = accuracy_score(y, predictions)
    return acc

#Show GBT trees. Last one from each block
def showTree(model, blockSize):
    for i in range(int(maxNumberOfTrees/blockSize)):
        plot_tree(model, num_trees=i*blockSize+(blockSize-1))
        plt.show()