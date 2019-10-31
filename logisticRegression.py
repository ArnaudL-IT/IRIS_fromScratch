import matplotlib.pyplot as plt
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

def costFunction(X, y, theta, regCoef):
    n_examples = np.size(X, 0)
    cost = - 1./n_examples * (y @ np.log(sigmoid(X @ theta)) + (1-y) @ np.log(1 - sigmoid(X @ theta)))
    costRegTerm = regCoef * theta[1::] @ theta[1::] / (2. * n_examples)
    cost += costRegTerm
    grad = 1./(2.*n_examples) * np.transpose(X) @ (sigmoid(X @ theta) - y)
    gradRegTerm = regCoef * theta / n_examples
    gradRegTerm[0] = 0
    grad += gradRegTerm
    return cost, grad

def update_theta(theta: np.array, grad: np.array, learning_rate: float, regCoef: float)->np.array:
    theta = theta - learning_rate * grad
    return theta

def lrPredict(theta: np.array, X: np.array)->np.array:
    proba = sigmoid(X @ theta)
    print(proba[0:5, :])
    idx = np.argmax(proba, axis = 1)
    predictions = np.zeros(proba.shape)
    for i in range(np.size(proba, 0)):
        for j in range(np.size(proba, 1)):
            if proba[i, j] == np.max(proba[i, :]): predictions [i, j] = 1
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
max_iterations = 10000
learning_rate = 0.01
regCoef = 1

# Initialize a n_features+1 x n_categories numpy array of uniformly distributed random parameters
theta = np.random.uniform(size=((n_features+1, n_categories))) * 2. - 1.

xs = np.arange(-10, 10, 1)
sig_func = sigmoid(xs)
plt.plot(xs, sig_func, 'r-')
plt.grid(True)
plt.title('Sigmoid function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

# Apply the One-vs-All algorithm
for i in range(n_categories):

    cost_to_plot = []

    for j in range(max_iterations):
        cost, grad = costFunction(X_train, y_train[:, i], theta[:, i], regCoef)
        theta[:, i] = update_theta(theta[:, i], grad, learning_rate, regCoef)
        cost_to_plot.append(cost)

    plt.plot(range(max_iterations), cost_to_plot, 'g-', label = 'cost')
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.show()

# Predict the category of each flower of the test set
predictions = lrPredict(theta, X_test)
print(predictions[0:5, :])
print(y_test[0:5, :])
# Compute the accuracy of the model
accuracy = np.mean([p == l for p, l in zip(predictions, y_test)])
print("Accuracy = {}".format(accuracy))
