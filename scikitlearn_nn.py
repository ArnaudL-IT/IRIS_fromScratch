from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Import the data into a Pandas DataFrame
with open('./data/iris.dat') as file:
    df = pd.read_csv(file, header = None)

# Give meaningful names to the columns of the data frame
feature_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Category']
df.columns = feature_names

# Split the data into a numpy array of features and labels
X = df.iloc[:, 0:4].to_numpy(copy = True)

# Encode the 3 string labels into an array of integers to use with the MLPClassifier
le = LabelEncoder()
le.fit(df['Category'].unique())
y = le.transform(df['Category'])

# Split the data-set into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

# Scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the Multi-Layer Percepron classifier from Scikit-learn
clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (5,), random_state=1)
clf.fit(X_train, y_train)

# Use the trained model to fit the test data-set
predictions = clf.predict(X_test)

# Compute the accuracy of the model
accuracy = np.mean([p == l for p, l in zip(predictions, y_test)])
print("Accuracy = {}".format(accuracy))
