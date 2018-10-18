# Iris_Flower_Prediction_A_ML_Tutorial
Here in this tutorial I explain and show how to get started with Machine Learning by applying supervised learning on UCI's famous Iris dataset.

If you are a machine learning beginner and looking to finally get started using Python, this tutorial was designed for you.

Let’s get started!

## Downloading, Installing and Starting Python SciPy

You will need to download and install below libraries for this project:
```
	1.scipy
	2.numpy
	3.pandas
	4.sklearn
```

I will not be covering how to install all of these libraries but [scipy page](https://www.scipy.org/install.html) shows it how to do.

### Check if all libraries are working good

Just write below lines in the python IDE to check if all work well:
```
import numpy
import pandas
import sklearn

```

## Load The Data

We are going to use the iris flowers dataset. This dataset is famous because it is used as the “hello world” dataset in machine learning and statistics by pretty much everyone.

The dataset contains 150 observations of iris flowers. There are four columns of measurements of the flowers in centimeters. The fifth column is the species of the flower observed. All observed flowers belong to one of three species.

In this step, we are going to load the iris data from CSV file URL. As UCI have been modifying few things on their website, so I have taken it from [Jason Brownlee](https://machinelearningmastery.com/about/) GitHub repository.

Here is the UCI official [Iris dataset](https://archive.ics.uci.edu/ml/datasets/Iris).

When you will see the code of this tutorial for loading the data, you may feel a bit confused but you can download the csv and just import it as:

```
data = pandas.read_csv('iris.csv')
```

## Evaluate Some Algorithms

Now it is time to create some models of the data and estimate their accuracy on unseen data.

Here is what we are going to cover in this step:
	1. Separate out a validation dataset.
	2. Set-up the test harness to use 10-fold cross-validation.
	3. Build 5 different models to predict species from flower measurements.
	4. Select the best model.

### Create a Validation Dataset

We need to know that the model we created is any good.

Later, we will use statistical methods to estimate the accuracy of the models that we create on unseen data. We also want a more concrete estimate of the accuracy of the best model on unseen data by evaluating it on actual unseen data.

That is, we are going to hold back some data that the algorithms will not get to see and we will use this data to get a second and independent idea of how accurate the best model might actually be.

We will split the loaded dataset into two, 80% of which we will use to train our models and 20% that we will hold back as a validation dataset.

### Build Models

We don’t know which algorithms would be good on this problem or what configurations to use. We get an idea from the plots that some of the classes are partially linearly separable in some dimensions, so we are expecting generally good results.

Let’s evaluate 6 different algorithms:

	1. Logistic Regression (LR)
	2. Linear Discriminant Analysis (LDA)
	3. K-Nearest Neighbors (KNN)
	4. Classification and Regression Trees (CART)
	5. Gaussian Naive Bayes (NB)
	6. Support Vector Machines (SVM)

This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB, and SVM) algorithms. We reset the random number seed before each run to ensure that the evaluation of each algorithm is performed using exactly the same data splits. It ensures the results are directly comparable.

## Predictions

In our tutorial KNN algorithm was the most accurate model that we tested. Now we want to get an idea of the accuracy of the model on our validation set.

This will give us an independent final check on the accuracy of the best model. It is valuable to keep a validation set just in case you made a slip during training, such as overfitting to the training set or a data leak. Both will result in an overly optimistic result.

We can run the KNN model directly on the validation set and summarize the results as a final accuracy score, a confusion matrix, and a classification report.

We can see in the tutorial that the accuracy is 0.9 or 90%. The confusion matrix provides an indication of the three errors made. Finally, the classification report provides a breakdown of each class by precision, recall, f1-score and support showing excellent results (granted the validation dataset was small).


## Summary

You do not need to understand everything. (at least not right now) Your goal is to run through the tutorial end-to-end and get a result. You do not need to understand everything on the first pass. Make heavy use of the help(“FunctionName”) help syntax in Python to learn about all of the functions that you’re using.
