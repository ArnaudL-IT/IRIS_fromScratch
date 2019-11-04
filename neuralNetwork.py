from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def split_data(data: np.array, labels: np.array, fraction = 0.7):
    train_size = (int)(np.size(data[:, 0])*fraction)
    test_size = np.size(data)-train_size

    # Make a list of randomized indices so the training and test sets have a mix of each category
    IDX = np.arange(150)
    np.random.shuffle(IDX)

    idx_train = IDX[0:train_size]
    idx_test = IDX[train_size:150]

    X_train = data[idx_train, :]
    y_train = labels[idx_train, :]
    X_test = data[idx_test, :]
    y_test = labels[idx_test, :]

    return X_train, X_test, y_train, y_test, idx_test

def sigmoid(X):
    return 1./(1.+np.exp(-X))

def cost_NN(theta1: np.array, theta2: np.array, X_train: np.array, y_train: np.array, hidden_layers: int)->np.array:
    n_examples = np.size(X_train, 0)

    A2 = np.concatenate((np.ones((n_examples, hidden_layers)), sigmoid(X_train @ theta1)), axis = 1)
    A3 = sigmoid(A2 @ theta2)

    cost = -1./ n_examples * np.sum(y @ np.transpose(np.log(A3)) + (1-y) @ np.transpose(np.log(1 - A3)))
    regCost = regCoef / (2. * n_examples) * (np.sum(np.square(theta1[1::,:])) + np.sum(np.square(theta2[1::,:])))
    cost += regCost

    delta_3 = A3 - y_train
    delta_2 = delta_3 @ np.transpose(theta2[1::, :]) * sigmoid(X_train @ theta1) * (1 - sigmoid(X_train @ theta1))
    grad_2 = np.transpose(A2) @ delta_3 / n_examples
    grad_1 = np.transpose(X_train) @ delta_2 / n_examples
    regGrad_2 = np.concatenate((np.zeros((1, 3)), regCoef / n_examples * theta2[1::, :]))
    regGrad_1 = np.concatenate((np.zeros((1, 4)), regCoef / n_examples * theta1[1::, :]))
    grad_2 += regGrad_2
    grad_1 += regGrad_1

    return cost, grad_1, grad_2

def update_param(theta1: np.array, theta2: np.array, grad_1: np.array, grad_2: np.array, learning_rate: float, regCoef: float):
    theta1 -= learning_rate * grad_1
    theta2 -= learning_rate * grad_2
    return theta1, theta2

def train_NN(theta1: np.array, theta2: np.array, X_train: np.array, y_train: np.array, hidden_layers: int, learning_rate: float, regCoef: float)->np.array:
    n_categories = np.size(y_train, axis = 1)
    max_iterations = 10000

    for i in range(n_categories):
        cost_to_print = []
        for j in range(max_iterations):
            cost, grad_1, grad_2 = cost_NN(theta1, theta2, X_train, y_train, regCoef)
            theta1, theta2 = update_param(theta1, theta2, grad_1, grad_2, learning_rate, regCoef)
            cost_to_print.append(cost)
        plt.plot(range(max_iterations), cost_to_print, 'r-')
        plt.show()

    return theta1, theta2

def nnPredict(theta1: np.array, theta2: np.array, X: np.array, hidden_layers: int)->np.array:
    n_examples = np.size(X, axis = 0)
    A2 = np.concatenate((np.ones((n_examples, hidden_layers)), sigmoid(X @ theta1)), axis = 1)
    A3 = sigmoid(A2 @ theta2)

    idx = np.argmax(A3, axis = 1)
    predictions = np.zeros(A3.shape)
    for i in range(np.size(A3, 0)):
        for j in range(np.size(A3, 1)):
            if A3[i, j] == np.max(A3[i, :]): predictions [i, j] = 1

    return predictions

# Import the data into a Pandas DataFrame
with open('./data/iris.dat') as file:
    df = pd.read_csv(file, header = None)

# Give meaningful names to the columns of the data frame
feature_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Category']
df.columns = feature_names

# Print some info about the data
print(df.head())
print(df.describe())
print(df.info())

# Split the data into a numpy array of features and labels
X = df.iloc[:, 0:4].to_numpy(copy = True)

n_categories = df['Category'].unique().size # Counts the number of uniques features

print("Number of categories found: {}".format(n_categories))
print(df['Category'].unique())

y = pd.get_dummies(df['Category']).to_numpy(copy = True)

# Set some useful variables
n_examples = np.size(X, 0)
n_features = np.size(X, 1)
n_categories = np.size(y, 1)

# Add ones to the data matrix for the intercept term
X = np.hstack((np.ones((n_examples, 1)), X))

# Split the data-set into training and test sets
X_train, X_test, y_train, y_test, idx_test = split_data(X, y, 0.6)

# Set the parameters of the model
learning_rate = 0.001
regCoef = 1

hidden_layers = 1
hidden_units_1 = 4

# Randomly initialize parameters of the model
theta1 = np.random.uniform(size=((n_features + 1, hidden_units_1))) * 2. - 1.
theta2 = np.random.uniform(size=((hidden_units_1 + 1, n_categories))) * 2. -1.

# Run forward and backward propagation to get the optimum set of parameters using the training set
theta1, theta2 = train_NN(theta1, theta2, X_train, y_train, hidden_layers, learning_rate, regCoef)

# Apply the parameters to the entries of the test set
predictions = nnPredict(theta1, theta2, X_test, hidden_layers)
print(predictions[0:5, :])
print(y_test[0:5, :])

# Compute accuracy of the model
accuracy = np.mean([p == l for p, l in zip(predictions, y_test)])
print("Accuracy = {}".format(accuracy))
