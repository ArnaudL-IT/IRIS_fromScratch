import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_features(data: np.array)->None:
    """ For a given numpy array, plots the histogram for each column """
    n_rows = np.size(data, 0)
    n_cols = np.size(data, 1)
    for i in range(n_cols):
        plt.hist(data[:,i])
        plt.show()

def study_correlation(data: np.array, labels: np.array)->None:
    """ Make a scatter plot of each pair of features """
    n_rows = np.size(data, 0)
    n_cols = np.size(data, 1)

    fig, ax = plt.subplots(n_cols, n_cols)

    for i in range(n_cols):
        for j in range(n_cols):
            if i != j: ax[i][j].scatter(data[:,j], data[:,i], c = labels)
            else: ax[i][j].annotate("series " + str(i), (0.5, 0.5), xycoords = 'axes fraction', ha = "center", va = "center")

            if i < n_cols-1: ax[i][j].xaxis.set_visible(False)
            if j > 0: ax[i][j].yaxis.set_visible(False)

    ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
    ax[0][0].set_ylim(ax[0][1].get_ylim())

    plt.show()
    plt.close()

    #print("Correlation between features {} and {} is {}".format(1, 2, 3))

def get_distances(data_scaled: np.array, centroids: np.array)->np.array:
    n_examples = np.size(data_scaled, 0)
    n_clusters = np.size(centroids, 0)
    distances = np.zeros((n_examples, n_clusters))
    for i  in range(n_examples):
        for j in range(n_clusters):
            distances[i][j] = np.sqrt(np.sum(np.square(data_scaled[i,:] - centroids[j])))

    return distances

def update_centroids(data_scaled: np.array, centroid_idx: np.array, n_examples: int)->np.array:
    IDX = np.zeros((n_examples, n_clusters))

    for i in range(n_examples):
        IDX[i, centroid_idx[i]] = 1

    weights = np.sum(IDX, axis = 0)
    centroids = np.dot(np.transpose(IDX), data_scaled)

    for i in range(n_clusters):
        centroids[i, :] /= weights[i]

    return centroids

def get_displacement(old_centroids: np.array, centroids: np.array)->float:
    displacement = np.sqrt(np.sum(np.square(old_centroids - centroids)))
    return displacement

def convert_predictions(predictions: np.array)->np.array:
    dict_indices = {}
    count0, count1, count2 = 0, 0, 0

    for i in range(50):
        if predictions[i] == 0: count0 += 1
        elif predictions[i] == 1: count1 += 1
        elif predictions[i] == 2: count2 +=1
        else: print("HUUUUUUUUUUGE problem")

    dict_indices[np.argmax([count0, count1, count2])] = [1, 0, 0]

    count0, count1, count2 = 0, 0, 0
    for i in range(50,100):
        if predictions[i] == 0: count0 += 1
        elif predictions[i] == 1: count1 += 1
        elif predictions[i] == 2: count2 +=1
        else: print("HUUUUUUUUUUGE problem")

    dict_indices[np.argmax([count0, count1, count2])] = [0, 1, 0]

    count0, count1, count2 = 0, 0, 0
    for i in range(100,150):
        if predictions[i] == 0: count0 += 1
        elif predictions[i] == 1: count1 += 1
        elif predictions[i] == 2: count2 +=1
        else: print("HUUUUUUUUUUGE problem")

    dict_indices[np.argmax([count0, count1, count2])] = [0, 0, 1]

#    if np.array(list_indices).unique.size != 3: print("ANOTHER PROBLEM!!")

    result = np.zeros((np.size(predictions, 0), 3))

    for i in range(150):
        result[i] = dict_indices[predictions[i]]

    return result

def train_KMean(data: np.array, labels: np.array, n_clusters: int)->None:
    """ Trains the K Mean algorithm """
    n_examples = np.size(data, 0)
    n_features = np.size(data, 1)

    # Scale the data so that Euclidian distance makes sense
    means = np.mean(data, axis = 0)
    stddevs = np.std(data, axis = 0, ddof = 1)

    #print(means)
    #print(stddevs)

    data_scaled = np.zeros((n_examples, n_features))

    for i in range(n_features):
        data_scaled[:, i] = (data[:,i] - means[i]) / stddevs[i]

    study_correlation(data_scaled)

    # Initialize the centroids
    idx = np.random.randint(n_examples, size = n_clusters)
    centroids = data_scaled[idx, :]

    counter = 0

    while True:

        distances = np.array([[np.sqrt(np.sum(np.square(example-centroid))) for centroid in centroids] for example in data_scaled])
        centroid_idx = np.argmin(distances, axis = 1)
        old_centroids = centroids
        centroids = update_centroids(data_scaled, centroid_idx, n_examples)
        #displacement = get_displacement(old_centroids, centroids)
        displacement = np.linalg.norm(np.array([old - new for old, new in zip(old_centroids, centroids)]))

        #assert np.linalg.norm(np.array([old - new for old, new in zip([1, 2, 3, 4], [5, 6, 7, 8])])) == 8

        if counter == 0:
#            print("Initial displacement = {}".format(displacement))
            initial_displacement = displacement

        counter += 1

        if displacement < (initial_displacement / 10000): break

    #print("Total number of loops before ending : {}".format(counter))
    converted_predictions = convert_predictions(centroid_idx)
    accuracy = np.mean([p == l for p, l in zip(converted_predictions, labels)])
    print("Accuracy = {}".format(accuracy))

    pass

def split_data(data: np.array, labels: np.array, fraction = 0.7):
    train_size = (int)(np.size(data[:, 0])*fraction)
    test_size = np.size(data)-train_size

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

def costFunction(X, y, theta):
    n_examples = np.size(X, 0)
    cost = 1./n_examples * (-y @ np.log(sigmoid(X @ theta)) - (1-y) @ np.log(1 - sigmoid(X @ theta)))
    grad = 1./(2.*n_examples) * np.transpose(X) @ (sigmoid(X @ theta) - y)
    return cost, grad

def update_theta(theta: np.array, grad: np.array, learning_rate: float)->np.array:
    theta = theta - learning_rate * grad
    return theta

def lrPredict(theta: np.array, X: np.array)->np.array:
    proba = sigmoid(X @ theta)
    print(proba[0:5, :])
    idx = np.argmax(proba, axis = 1)
    predictions = np.zeros(proba.shape)
    predictions[:, idx] = 1
    return predictions

def train_logisticRegression(data: np.array, labels: np.array)->None:
    """ Trains and estimate the performances of a logistic regression algorithm """

    n_examples = np.size(data, 0)
    n_features = np.size(data, 1)
    n_categories = np.size(labels, 1)

    data = np.hstack((np.ones((n_examples, 1)), data))

    print(data[0:5, :])

    X_train, X_test, y_train, y_test, idx_test = split_data(data, labels, 0.7)

    convergence_goal = 1e-3
    learning_rate = 0.01

    theta = np.random.uniform(size=((n_features+1, n_categories)))

    for i in range(n_categories):

        cost_var = 1

        previous_cost = 1e6
        iterations = 0
        cost_to_plot = []

        while cost_var > convergence_goal:
            iterations += 1
            cost, grad = costFunction(X_train, y_train[:, i], theta[:, i])
            theta[:, i] = update_theta(theta[:, i], grad, learning_rate)
            cost_var = previous_cost - cost
            previous_cost = cost
            if iterations == 1: cost_var = 1
            cost_to_plot.append(cost)
    #        print(cost)

        plt.plot(range(iterations), cost_to_plot, 'g-', label = 'cost')
        plt.xlabel('iterations')
        plt.ylabel('cost')
    #    plt.show()

    predictions = lrPredict(theta, X_test)

    print(predictions[0:5, :])
    print(y_test[0:5, :])

    accuracy = np.mean([p == l for p, l in zip(predictions, y_test)])
    print("Accuracy = {}".format(accuracy))

    pass

def train_SVM(data: np.array, labels: np.array)->None:
    """ Trains and estimate the performances of a SVM algorithm """
    print("SVM is not implemented yet!")

#####################################
#                                   #
#   Here starts the main program    #
#                                   #
#####################################
DEBUG = True

with open('./data/iris.dat') as file:
    df = pd.read_csv(file, header = None)

# Give meaningful names to the columns of the data frame
feature_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Category']
df.columns = feature_names

if DEBUG: print(df.head())
if DEBUG: print(df['Category'][45:55])

# Get some statistics on the data
print(df.describe())
print(df.info())

# Split the data into 3 sets (train, CV, test) and load into numpy arrays
X = df.iloc[:, 0:4].to_numpy(copy = True)


if DEBUG: print(X[45:55,:])
n_categories = df['Category'].unique().size # Counts the number of uniques features

print("Number of categories found: {}".format(n_categories))
print(df['Category'].unique())

# Get the labels into a 2D numpy array of 0 and 1's
y = pd.get_dummies(df['Category']).to_numpy(copy = True)

if DEBUG: print(y[45:55,:])

# DATA VISUALIZATION

if DEBUG: plot_features(X)

# Look at the correlation matrix and the scatter plot associated since there is only a few features

if DEBUG: study_correlation(X, y)

# Train K-Mean algorithm

n_clusters = n_categories   # We know exactly the number of possible categories

#for k in range(100):
#    train_KMean(X, y, n_clusters)

# Train logistic regression for multiclass classification

train_logisticRegression(X, y)

# Train SVM

train_SVM(X, y)
