Predicting Iris flower species using decision tree,Naive_bayes,support vector machine(linear,rbf kernels).

In iris_problem.ipynb:
This file is for understanding the features, target_name, target, data. Three examples are taken from the iris data and from iris target and stored in test data and test target, rest are stored in train_data and train_target.Decision tree classifier is used to train. Then these three test data are used to predict, whcih matches exactly with test target.

In iris_2.ipynb:
This file is same as iris_problem.ipynb but instead of using only 3 examples, i have used a random function to generate 30 numbers between 1-150 which are used for testing and rest for trainng. To check the accuracy i have imported accuracy_score from sklearn.metrics, here the accuracy is 96.6%

Used naive_bayes and support vector machine with linear and rbf kernel,but the accuracy is almost same in all cases as data set is small.
