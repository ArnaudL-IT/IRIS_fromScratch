# IRIS_fromScratch

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

Note: The "requirements.txt" file contains all necessary libraries to run the code. This file was created using the pip freeze command. Alll the libraries can be installed (in a virtual environment for example) by using:

```bash
pip install requirements.txt
```

**1. Data visualization**
In the first part of the code, the data-set is loaded into a pandas DataFrame from a CSV file. Some basic statistics is performed to get an idea of what the data contains.
Then, an array of 4x4 scatter plots is created using matplotlob to visualize the correlation bewtween each features 2 by 2. In a later version, the dots will be given colors according to their label (i.e. to which specie of Iris they belong).

**2. K-Mean algorithm**
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

**3. Logistic regression**
The second algorithm is an example of a logistic regression. This being a supervised learning algorithm requires to split the data-set into at least 2 subsets(training set and test set). Ideally, 3 subsets should be defined, including a cross-validation set used to tune the parameters of the model. The data-set being small and the number of free parameters as well, no cross-validation set is used here.

**4. Support Vector Machine**
