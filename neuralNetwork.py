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

hidden_layers = 1
hidden_units_1 = 4

# Randomly initialize parameters of the model
theta1 = np.random.uniform(size=((n_features + 1, hidden_units_1))) * 2. - 1.
theta2 = np.random.uniform(size=((hidden_units_1 + 1, n_categories)))

# Run forward and backward propagation to get the optimum set of parameters using the training set

# Apply the parameters to the entries of the test set

# Compute accuracy of the model
