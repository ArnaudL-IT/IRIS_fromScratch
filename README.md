# IRIS_fromScratch

__ONGOING PROJECT__

The Iris data-set is one of the simpler machine learning problem to study.
The data-set itself is made of 150 example of 3 different species of Iris flowers:
  - Iris setosa
  - Iris versicolor
  - Iris virginica
  
For each example, 4 features are given:
 - Sepal length
 - Sepal width
 - Petal length
 - Petal width
   
This project could easily be done using ML libraries but the goal here is to practice Python coding (assertion, type annotation, List comprehension, functional programming), the use of scientific libraries (numpy, pandas, matplotlib) and understand the maths behind each algorithms instead of simply feeding a few parameters to a method that will solve the problem for us.
As a consequence, some parts of the code are not implemented in the most condensed way or even the most efficient way but rather make use of one of the features mentionned above.

Note: The "requirements.txt" file contains all necessary libraries to run the code. This file was created using the pip freeze command. All the libraries can be installed (in a virtual environment for example) by using:

```bash
pip install -r requirements.txt
```
**Intro**<br />
Let's first have a more detailed look at the data before we start coding head down a ML algorithm. In IRIS_fromScratch.py, the data is loaded directly from the data file into a pandas' DataFrame. Altough pandas won't be used for much with such a simple data-set, the ```.describe()``` and ```.info()``` method used on the DataFrame are useful.
For example, ```.info()``` tells us that there is 150 non-null entries in each column and that the 4 first columns contain float64 type variables whereas the last one contains an object type (a string representing the specy of the iris).
Knowing that the data is clean, we can now proceed and run the ```.describe()``` method which gives us some statistics on our data. Even better is to combine it with the ```.groupby()``` method to get statistics on each feature for each category.
Looking at the mean and the standard deviation for the 4 features of each category, we can already tell that versicolor and the virginica species will be hard to dinstinguish from each other whereas the setosa has a well separated distributions for its petal length and width.

**1. Data visualization**<br />
So this was helpful but with more categories and features, it would be hard to draw conclusions so easily. A better way do proceed is to vizualize the data. Let's use matplotlib to represent the distributions associated with each features and for each category.
It is now clear that even the simple classification algorithm should be able to isolate the characteristics of an iris setosa and predict if a new flower belong to this category or not.

Then, an array of 4x4 scatter plots is created using matplotlob to visualize the correlation bewtween each features 2 by 2. In a later version, the dots will be given colors according to their label (i.e. to which specie of Iris they belong).

![Scatter plots](https://github.com/ArnaudL-IT/IRIS_fromScratch/blob/master/IMG/Figure_1.png)

**2. K-Mean algorithm**<br />
The first algorithm that I use to classify the data is an unsupervised learning algorithm. In this case, the labels are given for each examples but this is for educational purposes. Thus, the labels are not used until the end to calculate the accuracy of the model.
The algorithm is given 3 clusters since we know that there is only 3 different species of Iris in this data-set.
The main steps of this algorithm are:
 - Scaling the data. Always necessary for this type of algorithm using Euclidian distances between examples.
 - Initialize the 3 clusters to 3 randomly chosen examples.
 - Repeat the following steps until convergeance is reached:
   - Calculate the (Euclidian) distance of each example to each cluster.
   - Attribute each example to a cluster.
   - Calculate the center of mass of each cluster.
   - Move the clusters to their respective center of mass.

**3. Logistic regression**<br />
The second algorithm is an example of a logistic regression. This being a supervised learning algorithm requires to split the data-set into at least 2 subsets(training set and test set). Ideally, 3 subsets should be defined, including a cross-validation set used to tune the parameters of the model. The data-set being small and the number of free parameters as well, no cross-validation set is used here.

**4. Neural Network**<br />
A neural network is for me the most interesting algorithm to implement from scratch. It is a complex algorithm that requires a good understanding of linear algebra for an efficient vectorization. It is also a good oportunity to generalize the gradient descent from the logistic regression algorithm as well as the calculation of the cost function.

**5. Neural Network NOT from scratch (Scikit-learn)**<br />
Finaly, I wanted to use this simple data set to make my first dive into the Scikit-learn library. This library built on matplotlib, numpy and scipy provides all the tools that I developed above and so much more. The main reason to use it here is to start working with it, learn about it, see the different options availbale for a "Multi-Layer Perceptron Classifier" (name of the neural network class for one or more hidden layers) and also see what would be the improvement associated with the use of a highly optimized library.

Along with machine learning algorithms, Scikit-learn offers methods to (pre-)process the data such as encoding the labels of a list of examples (handy to switch from string labels, i.e. category names) to unique integers. There is also a StandardScaler which makes scaling the data a kid's game, and also a method to split the data into a train and test set simply by specifying the ratio of one of them and if the data set has to be shuffled before being split.

Altogether, the code ends up being much shorter and clearer and playing with the parameters of the model is simpler. One could now imagine looping over a given range of hidden units for the hidden layer or even over a range of hidden layers and find the best configuration for the NN.

Scikit-learn even propose toy data-sets to practice including the IRIS data-set but here I sticked to the same version I used for the other models.

**6. So what's next?**
Although the IRIS data set is pretty simple and not much more can be done in terms of data analysis, I would like to keep working on this project to implement/test new ideas. Below is a starting list of what I have in mind so far:
[]Regroup all the algorithms as options within the main program to run them all with the same command
[]Make plots of the evolution of the cost on a 3D plot and explore what matplotlib has to offer
[]Try to implement a class for a general neural network where the number of hidden layers and hidden units can be passed as arguments (much like the MLPClassifier from Scikit-learn)
